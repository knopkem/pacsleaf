//! Local viewer state and rendering flow for pacsleaf.

use crate::browser::{
    apply_window_geometry, capture_window_geometry, load_window_geometry, save_window_geometry,
    VIEWER_WINDOW_GEOMETRY_KEY,
};
use glam::{DQuat, DVec2, DVec3};
use leaf_core::domain::{InstanceInfo, SeriesInfo, SeriesUid, StudyUid};
use leaf_core::error::{LeafError, LeafResult};
use leaf_db::imagebox::Imagebox;
use leaf_dicom::metadata::read_instance_geometry;
use leaf_dicom::overlay::{load_overlays, OverlayBitmap};
use leaf_dicom::pixel::{decode_frame_with_window, frame_count};
use leaf_render::{
    lut::ColorLut, PreparedVolume, SlicePlane, SlicePreviewMode, SlicePreviewState,
    SliceProjectionMode, VolumeBlendMode, VolumePreviewImage, VolumePreviewRenderer,
    VolumeViewState,
};
use leaf_tools::measurement::{Measurement, MeasurementKind, MeasurementValue};
use lru::LruCache;
use slint::{ComponentHandle, ModelRc, VecModel};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::path::Path;
use std::rc::Rc;
use tracing::info;

const THUMB_SIZE: usize = 64;
const FRAME_CACHE_CAPACITY: usize = 32;
const DEFAULT_LUT_NAME: &str = "grayscale";
const LUT_PRESETS: [(&str, &str); 4] = [
    ("Gray", "grayscale"),
    ("Hot", "hot_iron"),
    ("Bone", "bone"),
    ("InvG", "grayscale_inverted"),
];
const OVERLAY_COLOR: [u8; 4] = [255, 196, 0, 255];

struct ViewerSession {
    viewer: slint::Weak<leaf_ui::StudyViewerWindow>,
    imagebox: Rc<Imagebox>,
    series: Vec<SeriesInfo>,
    instances_by_series: std::collections::HashMap<String, Vec<InstanceInfo>>,
    frames_by_series: HashMap<String, Vec<FrameRef>>,
    measurements_by_series: HashMap<String, Vec<Measurement>>,
    thumbnails_by_series: HashMap<String, slint::Image>,
    overlay_cache_by_file: HashMap<String, Vec<OverlayBitmap>>,
    active_series_uid: String,
    active_frame_index: usize,
    measurement_panel_visible: bool,
    volume_preview_active: bool,
    quad_viewport_active: bool,
    advanced_preview_mode: AdvancedPreviewMode,
    focused_quad_viewport: QuadViewportKind,
    active_tool: leaf_ui::ViewerTool,
    viewport_scale: f32,
    viewport_offset_x: f32,
    viewport_offset_y: f32,
    window_center: Option<f64>,
    window_width: Option<f64>,
    default_window_center: Option<f64>,
    default_window_width: Option<f64>,
    viewport_width: f32,
    viewport_height: f32,
    active_frame_width: u32,
    active_frame_height: u32,
    display_frame_width: u32,
    display_frame_height: u32,
    image_transform: ImageTransformState,
    active_lut_name: String,
    frame_cache: LruCache<FrameCacheKey, CachedFrame>,
    selected_measurement_id: Option<String>,
    draft_measurement: Option<DraftMeasurement>,
    handle_drag: Option<HandleDrag>,
    drag_state: Option<ViewportDragState>,
    volume_drag_state: Option<VolumeDragState>,
    volume_renderer: Option<VolumePreviewRenderer>,
    prepared_volumes_by_series: HashMap<String, PreparedVolume>,
    volume_view_state_by_series: HashMap<String, VolumeViewState>,
    slice_view_state_by_series: HashMap<String, SlicePreviewState>,
    quad_previews_by_kind: HashMap<QuadViewportKind, AdvancedViewportPreview>,
    quad_reference_lines_by_kind: HashMap<QuadViewportKind, Vec<QuadReferenceLineOverlay>>,
    quad_reference_hover: Option<QuadReferenceSelection>,
    quad_reference_drag: Option<QuadReferenceDrag>,
}

#[derive(Clone)]
struct FrameRef {
    file_path: String,
    frame_index: u32,
    image_orientation_patient: Option<[f64; 6]>,
}

#[derive(Clone, Copy)]
struct ViewportDragState {
    origin_x: f32,
    origin_y: f32,
    start_offset_x: f32,
    start_offset_y: f32,
    start_scale: f32,
    start_window_center: f64,
    start_window_width: f64,
}

#[derive(Clone, Copy)]
struct VolumeDragState {
    origin_x: f32,
    origin_y: f32,
    button: i32, // 0 = left, 2 = right
    start_view_state: VolumeViewState,
}

#[derive(Clone, Copy, Default)]
struct ImageTransformState {
    rotation_quarters: u8,
    flip_horizontal: bool,
    flip_vertical: bool,
    invert: bool,
}

impl ImageTransformState {
    fn is_rotated(self) -> bool {
        self.rotation_quarters % 4 != 0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum AdvancedPreviewMode {
    #[default]
    Dvr,
    Axial,
    Coronal,
    Sagittal,
}

impl AdvancedPreviewMode {
    fn label(self) -> &'static str {
        match self {
            Self::Dvr => "DVR",
            Self::Axial => "Ax",
            Self::Coronal => "Co",
            Self::Sagittal => "Sa",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Dvr => Self::Axial,
            Self::Axial => Self::Coronal,
            Self::Coronal => Self::Sagittal,
            Self::Sagittal => Self::Dvr,
        }
    }

    fn slice_mode(self) -> Option<SlicePreviewMode> {
        match self {
            Self::Dvr => None,
            Self::Axial => Some(SlicePreviewMode::Axial),
            Self::Coronal => Some(SlicePreviewMode::Coronal),
            Self::Sagittal => Some(SlicePreviewMode::Sagittal),
        }
    }

    fn is_dvr(self) -> bool {
        matches!(self, Self::Dvr)
    }

    fn quad_viewport(self) -> QuadViewportKind {
        match self {
            Self::Axial => QuadViewportKind::Axial,
            Self::Coronal => QuadViewportKind::Coronal,
            Self::Sagittal => QuadViewportKind::Sagittal,
            Self::Dvr => QuadViewportKind::Dvr,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum QuadViewportKind {
    Axial,
    Coronal,
    Sagittal,
    Dvr,
}

impl QuadViewportKind {
    const ALL: [Self; 4] = [Self::Axial, Self::Coronal, Self::Sagittal, Self::Dvr];

    fn from_index(index: i32) -> Option<Self> {
        match index {
            0 => Some(Self::Axial),
            1 => Some(Self::Coronal),
            2 => Some(Self::Sagittal),
            3 => Some(Self::Dvr),
            _ => None,
        }
    }

    fn index(self) -> i32 {
        match self {
            Self::Axial => 0,
            Self::Coronal => 1,
            Self::Sagittal => 2,
            Self::Dvr => 3,
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Axial => "Axial",
            Self::Coronal => "Coronal",
            Self::Sagittal => "Sagittal",
            Self::Dvr => "DVR",
        }
    }

    fn advanced_preview_mode(self) -> AdvancedPreviewMode {
        match self {
            Self::Axial => AdvancedPreviewMode::Axial,
            Self::Coronal => AdvancedPreviewMode::Coronal,
            Self::Sagittal => AdvancedPreviewMode::Sagittal,
            Self::Dvr => AdvancedPreviewMode::Dvr,
        }
    }

    fn slice_mode(self) -> Option<SlicePreviewMode> {
        self.advanced_preview_mode().slice_mode()
    }

    fn is_dvr(self) -> bool {
        matches!(self, Self::Dvr)
    }

    fn linked_mpr_views(self) -> [Self; 2] {
        match self {
            Self::Axial => [Self::Coronal, Self::Sagittal],
            Self::Coronal => [Self::Axial, Self::Sagittal],
            Self::Sagittal => [Self::Axial, Self::Coronal],
            Self::Dvr => [Self::Axial, Self::Coronal],
        }
    }
}

#[derive(Clone)]
struct AdvancedViewportPreview {
    width: u32,
    height: u32,
    image: slint::Image,
    info: String,
}

#[derive(Clone)]
struct QuadReferenceLineOverlay {
    commands: String,
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
    handle1_x: f32,
    handle1_y: f32,
    handle2_x: f32,
    handle2_y: f32,
    source_kind: QuadViewportKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuadReferenceTarget {
    Center,
    TranslateLine(QuadViewportKind),
    RotateLine(QuadViewportKind),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct QuadReferenceSelection {
    view: QuadViewportKind,
    target: QuadReferenceTarget,
}

#[derive(Clone, Copy)]
enum QuadReferenceDrag {
    Center {
        view: QuadViewportKind,
    },
    TranslateLine {
        view: QuadViewportKind,
        line_kind: QuadViewportKind,
        start_crosshair_world: DVec3,
        start_pointer_world: DVec3,
        line_normal: DVec3,
    },
    RotateLine {
        view: QuadViewportKind,
        line_kind: QuadViewportKind,
        start_angle_rad: f64,
        start_orientation: DQuat,
    },
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct FrameCacheKey {
    file_path: String,
    frame_index: u32,
    has_window_override: bool,
    window_center_bits: u64,
    window_width_bits: u64,
    lut_name: String,
    rotation_quarters: u8,
    flip_horizontal: bool,
    flip_vertical: bool,
    invert: bool,
}

#[derive(Clone)]
struct CachedFrame {
    source_width: u32,
    source_height: u32,
    display_width: u32,
    display_height: u32,
    rgba: Vec<u8>,
    window_center: f64,
    window_width: f64,
}

#[derive(Clone)]
enum DraftMeasurement {
    Line {
        start: DVec2,
        end: DVec2,
    },
    Angle {
        vertex: DVec2,
        arm1: DVec2,
        arm2: Option<DVec2>,
    },
    Rectangle {
        corner1: DVec2,
        corner2: DVec2,
    },
    Ellipse {
        center: DVec2,
        corner: DVec2,
    },
}

#[derive(Clone)]
struct HandleDrag {
    measurement_id: String,
    handle_index: usize,
}

#[derive(Clone, Copy)]
struct ViewportGeometry {
    image_origin_x: f32,
    image_origin_y: f32,
    image_width: f32,
    image_height: f32,
}

pub(crate) fn open_viewer_for_study(
    imagebox: &Rc<Imagebox>,
    study_uid: &str,
) -> LeafResult<leaf_ui::StudyViewerWindow> {
    let study_uid = StudyUid(study_uid.to_string());
    let study = imagebox
        .get_study(&study_uid)?
        .ok_or_else(|| LeafError::NoData(format!("Study {} not found", study_uid.0)))?;
    let mut series = imagebox.get_series_for_study(&study_uid)?;
    series.sort_by(|a, b| a.series_number.cmp(&b.series_number));

    let instances_by_series = series
        .iter()
        .map(|series| {
            let mut instances = imagebox
                .get_instances_for_series(&series.series_uid)
                .unwrap_or_default();
            sort_instances_for_stack(&mut instances);
            (series.series_uid.0.clone(), instances)
        })
        .collect::<std::collections::HashMap<_, _>>();
    let frames_by_series = build_frames_by_series(&instances_by_series);
    let mut measurements_by_series = HashMap::new();
    for series_info in &series {
        let uid = &series_info.series_uid.0;
        if let Ok(Some(json)) = imagebox.load_measurements(uid) {
            if let Ok(measurements) = serde_json::from_str::<Vec<Measurement>>(&json) {
                if !measurements.is_empty() {
                    measurements_by_series.insert(uid.clone(), measurements);
                }
            }
        }
    }

    // Load pre-generated thumbnails from DB, generate on-the-fly if missing
    let mut thumbnails_by_series = HashMap::new();
    for series_info in &series {
        let uid = &series_info.series_uid.0;
        // Try loading from DB first
        if let Ok(Some(rgba)) = imagebox.load_thumbnail(uid) {
            if rgba.len() == THUMB_SIZE * THUMB_SIZE * 4 {
                if let Ok(img) =
                    leaf_ui::image_from_rgba8(THUMB_SIZE as u32, THUMB_SIZE as u32, rgba)
                {
                    thumbnails_by_series.insert(uid.clone(), img);
                    continue;
                }
            }
        }
        // Generate lazily from middle frame if not in DB
        if let Some(instances) = instances_by_series.get(uid) {
            if let Some(rgba) = generate_thumbnail_rgba(instances) {
                if let Err(e) = imagebox.store_thumbnail(uid, &rgba) {
                    info!("Failed to cache thumbnail for {}: {}", uid, e);
                }
                if let Ok(img) =
                    leaf_ui::image_from_rgba8(THUMB_SIZE as u32, THUMB_SIZE as u32, rgba)
                {
                    thumbnails_by_series.insert(uid.clone(), img);
                }
            }
        }
    }

    let viewer =
        leaf_ui::StudyViewerWindow::new().map_err(|error| LeafError::Render(error.to_string()))?;
    if let Some(geometry) = load_window_geometry(imagebox, VIEWER_WINDOW_GEOMETRY_KEY)? {
        apply_window_geometry(viewer.window(), geometry);
    }
    viewer.set_patient_name(study.patient.patient_name.clone().into());
    viewer.set_connection_status("Local imagebox".into());
    viewer.set_active_tool(leaf_ui::ViewerTool::WindowLevel);
    viewer.set_volume_preview_active(false);
    viewer.set_quad_view_active(false);
    viewer.set_layout_label("1Up".into());
    viewer.set_focused_quad_viewport(AdvancedPreviewMode::default().quad_viewport().index());
    viewer.set_study_description(
        study
            .study_description
            .clone()
            .unwrap_or_else(|| "Untitled study".to_string())
            .into(),
    );
    viewer.set_measurement_panel_visible(false);

    let active_series_uid = series
        .first()
        .map(|series| series.series_uid.0.clone())
        .unwrap_or_default();
    let session = Rc::new(RefCell::new(ViewerSession {
        viewer: viewer.as_weak(),
        imagebox: imagebox.clone(),
        series,
        instances_by_series,
        frames_by_series,
        measurements_by_series,
        thumbnails_by_series,
        overlay_cache_by_file: HashMap::new(),
        active_series_uid,
        active_frame_index: 0,
        measurement_panel_visible: false,
        volume_preview_active: false,
        quad_viewport_active: false,
        advanced_preview_mode: AdvancedPreviewMode::default(),
        focused_quad_viewport: AdvancedPreviewMode::default().quad_viewport(),
        active_tool: leaf_ui::ViewerTool::WindowLevel,
        viewport_scale: 1.0,
        viewport_offset_x: 0.0,
        viewport_offset_y: 0.0,
        window_center: None,
        window_width: None,
        default_window_center: None,
        default_window_width: None,
        viewport_width: 0.0,
        viewport_height: 0.0,
        active_frame_width: 0,
        active_frame_height: 0,
        display_frame_width: 0,
        display_frame_height: 0,
        image_transform: ImageTransformState::default(),
        active_lut_name: DEFAULT_LUT_NAME.to_string(),
        frame_cache: LruCache::new(
            NonZeroUsize::new(FRAME_CACHE_CAPACITY).expect("frame cache capacity must be > 0"),
        ),
        selected_measurement_id: None,
        draft_measurement: None,
        handle_drag: None,
        drag_state: None,
        volume_drag_state: None,
        volume_renderer: None,
        prepared_volumes_by_series: HashMap::new(),
        volume_view_state_by_series: HashMap::new(),
        slice_view_state_by_series: HashMap::new(),
        quad_previews_by_kind: HashMap::new(),
        quad_reference_lines_by_kind: HashMap::new(),
        quad_reference_hover: None,
        quad_reference_drag: None,
    }));

    {
        let session_ref = session.borrow();
        update_series_model(&session_ref)?;
    }
    {
        let mut session_ref = session.borrow_mut();
        update_viewer_image(&mut session_ref)?;
    }
    {
        let session_ref = session.borrow();
        update_measurements_model(&session_ref)?;
        let _ = update_measurement_overlays(&session_ref);
    }

    let session_for_series = session.clone();
    viewer.on_series_selected(move |series_uid| {
        let mut session = session_for_series.borrow_mut();
        let preview_active = session.volume_preview_active;
        session.active_series_uid = series_uid.to_string();
        session.active_frame_index = 0;
        session.selected_measurement_id = None;
        session.draft_measurement = None;
        session.handle_drag = None;
        session.quad_reference_drag = None;
        reset_viewport_state(&mut session, true);
        let result = update_series_model(&session).and_then(|_| {
            if preview_active {
                render_or_show_volume_preview(&mut session)
            } else {
                update_viewer_image(&mut session)
                    .and_then(|_| update_measurements_model(&session))
                    .and_then(|_| update_measurement_overlays(&session))
            }
        });
        if let Err(error) = result {
            info!("Failed to switch series: {}", error);
        }
    });

    let session_for_tool = session.clone();
    viewer.on_tool_selected(move |tool| {
        let mut session = session_for_tool.borrow_mut();
        let had_draft = session.draft_measurement.is_some();
        session.active_tool = tool;
        session.drag_state = None;
        session.volume_drag_state = None;
        session.draft_measurement = None;
        session.handle_drag = None;
        session.quad_reference_drag = None;
        // Only rebuild measurement overlays when a draft measurement was cleared;
        // otherwise the overlays haven't changed and the Slint model rebuild is
        // pure overhead.
        if had_draft {
            if let Err(error) = apply_viewport_state(&session) {
                info!("Failed to switch tool: {}", error);
            }
        } else if let Some(viewer) = session.viewer.upgrade() {
            viewer.set_active_tool(session.active_tool);
        }
    });

    let session_for_rotate = session.clone();
    viewer.on_rotate_image(move || {
        let mut session = session_for_rotate.borrow_mut();
        if session.volume_preview_active {
            return;
        }
        session.image_transform.rotation_quarters =
            (session.image_transform.rotation_quarters + 1) % 4;
        if let Err(error) = update_viewer_image(&mut session) {
            info!("Failed to rotate image: {}", error);
        }
    });

    let session_for_flip_h = session.clone();
    viewer.on_flip_horizontal(move || {
        let mut session = session_for_flip_h.borrow_mut();
        if session.volume_preview_active {
            return;
        }
        session.image_transform.flip_horizontal = !session.image_transform.flip_horizontal;
        if let Err(error) = update_viewer_image(&mut session) {
            info!("Failed to flip image horizontally: {}", error);
        }
    });

    let session_for_flip_v = session.clone();
    viewer.on_flip_vertical(move || {
        let mut session = session_for_flip_v.borrow_mut();
        if session.volume_preview_active {
            return;
        }
        session.image_transform.flip_vertical = !session.image_transform.flip_vertical;
        if let Err(error) = update_viewer_image(&mut session) {
            info!("Failed to flip image vertically: {}", error);
        }
    });

    let session_for_invert = session.clone();
    viewer.on_invert_image(move || {
        let mut session = session_for_invert.borrow_mut();
        if session.volume_preview_active {
            return;
        }
        session.image_transform.invert = !session.image_transform.invert;
        if let Err(error) = update_viewer_image(&mut session) {
            info!("Failed to invert image: {}", error);
        }
    });

    let session_for_lut = session.clone();
    viewer.on_cycle_lut(move || {
        let mut session = session_for_lut.borrow_mut();
        if session.volume_preview_active {
            return;
        }
        session.active_lut_name = next_lut_name(&session.active_lut_name).to_string();
        if let Err(error) = update_viewer_image(&mut session) {
            info!("Failed to change LUT: {}", error);
        }
    });

    let session_for_scroll = session.clone();
    viewer.on_viewport_scroll(move |delta| {
        let mut session = session_for_scroll.borrow_mut();
        if session.volume_preview_active {
            if session.quad_viewport_active {
                let focused_quad_viewport = session.focused_quad_viewport;
                let result = if session.focused_quad_viewport.is_dvr() {
                    let zoom_factor = if delta > 0.0 {
                        1.1
                    } else if delta < 0.0 {
                        0.9
                    } else {
                        1.0
                    };
                    active_volume_view_state(&mut session).zoom_by(zoom_factor);
                    render_quad_single_preview(&mut session, QuadViewportKind::Dvr, false)
                } else {
                    adjust_quad_crosshair_by_scroll(&mut session, focused_quad_viewport, delta)
                        .and_then(|_| render_quad_mpr_previews(&mut session))
                };
                if let Err(error) = result {
                    info!("Failed to update quad viewport scroll: {}", error);
                }
                return;
            }
            if session.advanced_preview_mode.is_dvr() {
                let zoom_factor = if delta > 0.0 {
                    1.1
                } else if delta < 0.0 {
                    0.9
                } else {
                    1.0
                };
                active_volume_view_state(&mut session).zoom_by(zoom_factor);
            } else if let Some(slice_mode) = session.advanced_preview_mode.slice_mode() {
                let bounds_and_step = session
                    .prepared_volumes_by_series
                    .get(&session.active_series_uid)
                    .map(|prepared| {
                        (
                            prepared.world_bounds(),
                            prepared.slice_scroll_step(slice_mode),
                        )
                    });
                if let Some((bounds, step)) = bounds_and_step {
                    let delta_mm = if delta < 0.0 {
                        step
                    } else if delta > 0.0 {
                        -step
                    } else {
                        0.0
                    };
                    active_slice_view_state(&mut session).scroll_by(delta_mm, bounds);
                }
            }
            if let Err(error) = render_or_show_volume_preview(&mut session) {
                info!("Failed to update advanced preview: {}", error);
            }
            return;
        }
        session.draft_measurement = None;
        let step = if delta < 0.0 {
            1
        } else if delta > 0.0 {
            -1
        } else {
            0
        };
        let max_index = session
            .frames_by_series
            .get(&session.active_series_uid)
            .map(|frames| frames.len().saturating_sub(1))
            .unwrap_or(0);
        session.active_frame_index =
            (session.active_frame_index as i32 + step).clamp(0, max_index as i32) as usize;
        if let Err(error) = update_viewer_image(&mut session) {
            info!("Failed to scroll viewport: {}", error);
        }
    });

    let session_for_mouse_down = session.clone();
    viewer.on_viewport_mouse_down(move |x, y, viewport_width, viewport_height, button| {
        let mut session = session_for_mouse_down.borrow_mut();
        update_viewport_dimensions(&mut session, viewport_width, viewport_height);
        if session.volume_preview_active {
            if session.advanced_preview_mode.is_dvr() {
                session.drag_state = None;
                session.volume_drag_state = Some(VolumeDragState {
                    origin_x: x,
                    origin_y: y,
                    button,
                    start_view_state: *active_volume_view_state(&mut session),
                });
            } else {
                session.volume_drag_state = None;
                if button == 2 {
                    session.drag_state = None;
                    if let Some(world) = mpr_world_point_from_viewport(&session, x, y) {
                        active_slice_view_state(&mut session).set_crosshair_world(world);
                        if let Err(error) = render_or_show_volume_preview(&mut session) {
                            info!("Failed to update MPR crosshair: {}", error);
                        }
                    }
                    return;
                }
                let scalar_range = session
                    .prepared_volumes_by_series
                    .get(&session.active_series_uid)
                    .map(|prepared| prepared.scalar_range());
                session.drag_state = match (session.active_tool, scalar_range) {
                    (
                        leaf_ui::ViewerTool::WindowLevel
                        | leaf_ui::ViewerTool::Pan
                        | leaf_ui::ViewerTool::Zoom,
                        Some((scalar_min, scalar_max)),
                    ) => {
                        let (start_center, start_width) = active_slice_view_state(&mut session)
                            .transfer_window(scalar_min, scalar_max);
                        Some(ViewportDragState {
                            origin_x: x,
                            origin_y: y,
                            start_offset_x: session.viewport_offset_x,
                            start_offset_y: session.viewport_offset_y,
                            start_scale: session.viewport_scale,
                            start_window_center: start_center,
                            start_window_width: start_width,
                        })
                    }
                    _ => None,
                };
            }
            return;
        }
        match session.active_tool {
            leaf_ui::ViewerTool::WindowLevel
            | leaf_ui::ViewerTool::Pan
            | leaf_ui::ViewerTool::Zoom => {
                session.drag_state = Some(ViewportDragState {
                    origin_x: x,
                    origin_y: y,
                    start_offset_x: session.viewport_offset_x,
                    start_offset_y: session.viewport_offset_y,
                    start_scale: session.viewport_scale,
                    start_window_center: session.window_center.unwrap_or(0.0),
                    start_window_width: session.window_width.unwrap_or(1.0),
                });
            }
            leaf_ui::ViewerTool::Line
            | leaf_ui::ViewerTool::Angle
            | leaf_ui::ViewerTool::RectangleRoi
            | leaf_ui::ViewerTool::EllipseRoi => {
                session.drag_state = None;
                // If in angle phase 2, continue with arm2
                if matches!(
                    &session.draft_measurement,
                    Some(DraftMeasurement::Angle { arm2: Some(_), .. })
                ) {
                    if let Some(point) = viewport_to_image_point(&session, x, y, false) {
                        if let Some(DraftMeasurement::Angle { arm2, .. }) =
                            session.draft_measurement.as_mut()
                        {
                            *arm2 = Some(point);
                        }
                    }
                    return;
                }
                // Check for handle drag on existing measurement
                if let Some((measurement_id, handle_index)) = find_handle_at(&session, x, y) {
                    session.selected_measurement_id = Some(measurement_id.clone());
                    session.handle_drag = Some(HandleDrag {
                        measurement_id,
                        handle_index,
                    });
                    session.draft_measurement = None;
                    if let Err(error) = update_measurement_overlays(&session) {
                        info!("Failed to update overlays on handle select: {}", error);
                    }
                    return;
                }
                // Start new draft
                let Some(point) = viewport_to_image_point(&session, x, y, false) else {
                    return;
                };
                session.draft_measurement = Some(match session.active_tool {
                    leaf_ui::ViewerTool::Line => DraftMeasurement::Line {
                        start: point,
                        end: point,
                    },
                    leaf_ui::ViewerTool::Angle => DraftMeasurement::Angle {
                        vertex: point,
                        arm1: point,
                        arm2: None,
                    },
                    leaf_ui::ViewerTool::RectangleRoi => DraftMeasurement::Rectangle {
                        corner1: point,
                        corner2: point,
                    },
                    leaf_ui::ViewerTool::EllipseRoi => DraftMeasurement::Ellipse {
                        center: point,
                        corner: point,
                    },
                    _ => unreachable!(),
                });
                if let Err(error) = update_measurement_overlays(&session) {
                    info!("Failed to begin measurement: {}", error);
                }
            }
            _ => {
                session.drag_state = None;
                session.draft_measurement = None;
                session.handle_drag = None;
            }
        }
    });

    let session_for_mouse_up = session.clone();
    viewer.on_viewport_mouse_up(move |x, y, viewport_width, viewport_height| {
        let mut session = session_for_mouse_up.borrow_mut();
        update_viewport_dimensions(&mut session, viewport_width, viewport_height);
        if session.volume_preview_active {
            if session.advanced_preview_mode.is_dvr() {
                session.volume_drag_state = None;
                if let Err(error) = render_volume_preview(&mut session, false) {
                    info!("Failed to finalize volume preview interaction: {}", error);
                }
            } else {
                session.drag_state = None;
            }
            return;
        }

        // Finalize handle drag
        if session.handle_drag.is_some() {
            session.handle_drag = None;
            let result = update_measurements_model(&session)
                .and_then(|_| update_measurement_overlays(&session));
            if let Err(error) = result {
                info!("Failed to finalize handle drag: {}", error);
            }
            persist_measurements(&session);
            return;
        }

        match session.active_tool {
            leaf_ui::ViewerTool::Line
            | leaf_ui::ViewerTool::RectangleRoi
            | leaf_ui::ViewerTool::EllipseRoi => {
                if let Some(point) = viewport_to_image_point(&session, x, y, true) {
                    match session.draft_measurement.as_mut() {
                        Some(DraftMeasurement::Line { end, .. }) => *end = point,
                        Some(DraftMeasurement::Rectangle { corner2, .. }) => *corner2 = point,
                        Some(DraftMeasurement::Ellipse { corner, .. }) => *corner = point,
                        _ => {}
                    }
                }
                if let Err(error) = finalize_measurement(&mut session) {
                    info!("Failed to finalize measurement: {}", error);
                }
            }
            leaf_ui::ViewerTool::Angle => {
                let was_phase_2 = matches!(
                    &session.draft_measurement,
                    Some(DraftMeasurement::Angle { arm2: Some(_), .. })
                );
                if let Some(point) = viewport_to_image_point(&session, x, y, true) {
                    if let Some(DraftMeasurement::Angle { arm1, arm2, .. }) =
                        session.draft_measurement.as_mut()
                    {
                        if arm2.is_none() {
                            *arm1 = point;
                            *arm2 = Some(point);
                        } else {
                            *arm2 = Some(point);
                        }
                    }
                }
                if was_phase_2 {
                    if let Err(error) = finalize_measurement(&mut session) {
                        info!("Failed to finalize angle measurement: {}", error);
                    }
                } else if let Err(error) = update_measurement_overlays(&session) {
                    info!("Failed to update angle measurement: {}", error);
                }
            }
            _ => {
                session.drag_state = None;
            }
        }
    });

    let session_for_mouse_move = session.clone();
    viewer.on_viewport_mouse_move(move |x, y, viewport_width, viewport_height| {
        let mut session = session_for_mouse_move.borrow_mut();
        update_viewport_dimensions(&mut session, viewport_width, viewport_height);
        if session.volume_preview_active && session.advanced_preview_mode.is_dvr() {
            let Some(drag_state) = session.volume_drag_state else {
                return;
            };
            let dx = x - drag_state.origin_x;
            let dy = y - drag_state.origin_y;
            apply_volume_drag(
                &mut session,
                drag_state.start_view_state,
                dx,
                dy,
                drag_state.button,
            );
            if let Err(error) = render_volume_preview(&mut session, true) {
                info!("Failed to update volume preview interaction: {}", error);
            }
            return;
        }

        if session.volume_preview_active {
            let Some(drag_state) = session.drag_state else {
                return;
            };
            let dx = x - drag_state.origin_x;
            let dy = y - drag_state.origin_y;
            let result = match session.active_tool {
                leaf_ui::ViewerTool::Pan => {
                    session.viewport_offset_x = drag_state.start_offset_x + dx;
                    session.viewport_offset_y = drag_state.start_offset_y + dy;
                    apply_viewport_state(&session)
                }
                leaf_ui::ViewerTool::Zoom => {
                    let factor = (1.0 - dy * 0.01).max(0.1);
                    session.viewport_scale = (drag_state.start_scale * factor).clamp(0.25, 8.0);
                    apply_viewport_state(&session)
                }
                leaf_ui::ViewerTool::WindowLevel => {
                    let scalar_range = session
                        .prepared_volumes_by_series
                        .get(&session.active_series_uid)
                        .map(|prepared| prepared.scalar_range());
                    if let Some((scalar_min, scalar_max)) = scalar_range {
                        let sensitivity = (drag_state.start_window_width / 512.0).max(1.0);
                        let center = drag_state.start_window_center + dx as f64 * sensitivity;
                        let width =
                            (drag_state.start_window_width - dy as f64 * sensitivity).max(1.0);
                        active_slice_view_state(&mut session)
                            .set_transfer_window(center, width, scalar_min, scalar_max);
                        render_or_show_volume_preview(&mut session)
                    } else {
                        Ok(())
                    }
                }
                _ => Ok(()),
            };

            if let Err(error) = result {
                info!("Failed to update MPR interaction: {}", error);
            }
            return;
        }

        // Handle drag on existing measurement handle
        if let Some(handle_drag) = session.handle_drag.clone() {
            if let Some(point) = viewport_to_image_point(&session, x, y, true) {
                let series_uid = session.active_series_uid.clone();
                if let Some(measurements) = session.measurements_by_series.get_mut(&series_uid) {
                    if let Some(m) = measurements
                        .iter_mut()
                        .find(|m| m.id == handle_drag.measurement_id)
                    {
                        m.set_handle_position(handle_drag.handle_index, point);
                    }
                }
            }
            if let Err(error) = update_measurement_overlays(&session) {
                info!("Failed to update handle drag: {}", error);
            }
            return;
        }

        // Update draft measurement preview
        if session.draft_measurement.is_some() {
            if let Some(point) = viewport_to_image_point(&session, x, y, true) {
                match session.draft_measurement.as_mut() {
                    Some(DraftMeasurement::Line { end, .. }) => *end = point,
                    Some(DraftMeasurement::Angle { arm1, arm2, .. }) => {
                        if arm2.is_none() {
                            *arm1 = point;
                        } else {
                            *arm2 = Some(point);
                        }
                    }
                    Some(DraftMeasurement::Rectangle { corner2, .. }) => *corner2 = point,
                    Some(DraftMeasurement::Ellipse { corner, .. }) => *corner = point,
                    _ => {}
                }
            }
            if let Err(error) = update_measurement_overlays(&session) {
                info!("Failed to update measurement preview: {}", error);
            }
            return;
        }

        let Some(drag_state) = session.drag_state else {
            return;
        };

        let dx = x - drag_state.origin_x;
        let dy = y - drag_state.origin_y;

        let result = match session.active_tool {
            leaf_ui::ViewerTool::Pan => {
                session.viewport_offset_x = drag_state.start_offset_x + dx;
                session.viewport_offset_y = drag_state.start_offset_y + dy;
                apply_viewport_state(&session)
            }
            leaf_ui::ViewerTool::Zoom => {
                let factor = (1.0 - dy * 0.01).max(0.1);
                session.viewport_scale = (drag_state.start_scale * factor).clamp(0.25, 8.0);
                apply_viewport_state(&session)
            }
            leaf_ui::ViewerTool::WindowLevel => {
                let sensitivity = (drag_state.start_window_width / 512.0).max(1.0);
                session.window_center =
                    Some(drag_state.start_window_center + dx as f64 * sensitivity);
                session.window_width =
                    Some((drag_state.start_window_width - dy as f64 * sensitivity).max(1.0));
                update_viewer_image(&mut session)
            }
            _ => Ok(()),
        };

        if let Err(error) = result {
            info!("Failed to update viewport interaction: {}", error);
        }
    });

    let session_for_quad_mouse_down = session.clone();
    viewer.on_quad_viewport_mouse_down(
        move |index, x, y, viewport_width, viewport_height, button| {
            let Some(kind) = QuadViewportKind::from_index(index) else {
                return;
            };
            let mut session = session_for_quad_mouse_down.borrow_mut();
            set_selected_quad_viewport(&mut session, kind);
            session.quad_reference_drag = None;
            session.quad_reference_hover = None;
            if !session.volume_preview_active || !session.quad_viewport_active {
                let _ = apply_viewport_state(&session);
                return;
            }

            if kind.is_dvr() {
                session.drag_state = None;
                session.volume_drag_state = Some(VolumeDragState {
                    origin_x: x,
                    origin_y: y,
                    button,
                    start_view_state: *active_volume_view_state(&mut session),
                });
                let _ = apply_viewport_state(&session);
                return;
            }

            session.volume_drag_state = None;
            if button == 2 {
                if let Some(world) = quad_mpr_world_point_from_viewport(
                    &session,
                    kind,
                    x,
                    y,
                    viewport_width,
                    viewport_height,
                ) {
                    let bounds = session
                        .prepared_volumes_by_series
                        .get(&session.active_series_uid)
                        .map(|prepared| prepared.world_bounds());
                    if let (Some(slice_mode), Some(bounds)) = (kind.slice_mode(), bounds) {
                        let slice_state = active_slice_view_state(&mut session);
                        slice_state.set_mode(slice_mode);
                        slice_state.center_on_world(world, bounds);
                        if let Err(error) = render_quad_mpr_previews(&mut session) {
                            info!("Failed to update quad MPR crosshair: {}", error);
                        }
                    }
                }
                return;
            }

            if let Some(hit) =
                quad_reference_line_hit(&session, kind, x, y, viewport_width, viewport_height)
            {
                session.quad_reference_hover = Some(hit);
                session.drag_state = None;
                match hit.target {
                    QuadReferenceTarget::Center => {
                        session.quad_reference_drag =
                            Some(QuadReferenceDrag::Center { view: kind });
                        let _ = apply_viewport_state(&session);
                        return;
                    }
                    QuadReferenceTarget::RotateLine(line_kind) => {
                        if let Some(start_angle_rad) = quad_mpr_angle_from_viewport(
                            &session,
                            kind,
                            x,
                            y,
                            viewport_width,
                            viewport_height,
                        ) {
                            let start_orientation = session
                                .slice_view_state_by_series
                                .get(&session.active_series_uid)
                                .copied()
                                .unwrap_or_default()
                                .orientation;
                            session.quad_reference_drag = Some(QuadReferenceDrag::RotateLine {
                                view: kind,
                                line_kind,
                                start_angle_rad,
                                start_orientation,
                            });
                            let _ = apply_viewport_state(&session);
                            return;
                        }
                    }
                    QuadReferenceTarget::TranslateLine(line_kind) => {
                        let drag = session
                            .prepared_volumes_by_series
                            .get(&session.active_series_uid)
                            .and_then(|prepared| {
                                let bounds = prepared.world_bounds();
                                let current_state =
                                    quad_slice_view_state_for_kind(&session, prepared, kind)?;
                                let other_state =
                                    quad_slice_view_state_for_kind(&session, prepared, line_kind)?;
                                let start_pointer_world = quad_mpr_world_point_from_viewport(
                                    &session,
                                    kind,
                                    x,
                                    y,
                                    viewport_width,
                                    viewport_height,
                                )?;
                                let start_crosshair_world = current_state.crosshair_world(bounds);
                                let current_plane = current_state.slice_plane(bounds);
                                let other_plane = other_state.slice_plane(bounds);
                                let line_dir = current_plane.normal().cross(other_plane.normal());
                                if line_dir.length_squared() <= 1.0e-10 {
                                    return None;
                                }
                                let line_normal =
                                    current_plane.normal().cross(line_dir.normalize());
                                if line_normal.length_squared() <= 1.0e-10 {
                                    return None;
                                }
                                Some(QuadReferenceDrag::TranslateLine {
                                    view: kind,
                                    line_kind,
                                    start_crosshair_world,
                                    start_pointer_world,
                                    line_normal: line_normal.normalize(),
                                })
                            });
                        if let Some(drag) = drag {
                            session.quad_reference_drag = Some(drag);
                            let _ = apply_viewport_state(&session);
                            return;
                        }
                    }
                }
            }
            if session.quad_reference_hover.is_some() {
                session.quad_reference_hover = None;
                let _ = apply_viewport_state(&session);
            }

            let scalar_range = session
                .prepared_volumes_by_series
                .get(&session.active_series_uid)
                .map(|prepared| prepared.scalar_range());
            session.drag_state = match (session.active_tool, scalar_range) {
                (leaf_ui::ViewerTool::WindowLevel, Some((scalar_min, scalar_max))) => {
                    let slice_mode = kind.slice_mode().unwrap_or_default();
                    let bounds = session
                        .prepared_volumes_by_series
                        .get(&session.active_series_uid)
                        .map(|prepared| prepared.world_bounds());
                    let slice_state = active_slice_view_state(&mut session);
                    if slice_state.mode != slice_mode {
                        slice_state.set_mode(slice_mode);
                    }
                    if let Some(bounds) = bounds {
                        slice_state.center_on_crosshair(bounds);
                    }
                    let (start_center, start_width) =
                        slice_state.transfer_window(scalar_min, scalar_max);
                    Some(ViewportDragState {
                        origin_x: x,
                        origin_y: y,
                        start_offset_x: 0.0,
                        start_offset_y: 0.0,
                        start_scale: 1.0,
                        start_window_center: start_center,
                        start_window_width: start_width,
                    })
                }
                _ => None,
            };
            let _ = apply_viewport_state(&session);
        },
    );

    let session_for_quad_mouse_up = session.clone();
    viewer.on_quad_viewport_mouse_up(move |index, x, y, viewport_width, viewport_height| {
        let Some(kind) = QuadViewportKind::from_index(index) else {
            return;
        };
        let mut session = session_for_quad_mouse_up.borrow_mut();
        set_selected_quad_viewport(&mut session, kind);
        if !session.volume_preview_active || !session.quad_viewport_active {
            return;
        }
        session.quad_reference_drag = None;
        session.quad_reference_hover = if kind.is_dvr() {
            None
        } else {
            quad_reference_line_hit(&session, kind, x, y, viewport_width, viewport_height)
        };
        if kind.is_dvr() {
            session.volume_drag_state = None;
            if let Err(error) =
                render_quad_single_preview(&mut session, QuadViewportKind::Dvr, false)
            {
                info!("Failed to finalize quad DVR interaction: {}", error);
            }
        } else {
            session.drag_state = None;
        }
    });

    let session_for_quad_mouse_move = session.clone();
    viewer.on_quad_viewport_mouse_move(move |index, x, y, viewport_width, viewport_height| {
        let Some(kind) = QuadViewportKind::from_index(index) else {
            return;
        };
        let mut session = session_for_quad_mouse_move.borrow_mut();
        if !session.volume_preview_active || !session.quad_viewport_active {
            return;
        }

        if kind.is_dvr() {
            let Some(drag_state) = session.volume_drag_state else {
                return;
            };
            let dx = x - drag_state.origin_x;
            let dy = y - drag_state.origin_y;
            apply_volume_drag(
                &mut session,
                drag_state.start_view_state,
                dx,
                dy,
                drag_state.button,
            );
            if let Err(error) =
                render_quad_single_preview(&mut session, QuadViewportKind::Dvr, true)
            {
                info!("Failed to update quad DVR interaction: {}", error);
            }
            return;
        }

        if let Some(reference_drag) = session.quad_reference_drag {
            let result = match reference_drag {
                QuadReferenceDrag::Center { view } => {
                    if view != kind {
                        return;
                    }
                    if let Some(world) = quad_mpr_world_point_from_viewport(
                        &session,
                        kind,
                        x,
                        y,
                        viewport_width,
                        viewport_height,
                    ) {
                        let bounds = session
                            .prepared_volumes_by_series
                            .get(&session.active_series_uid)
                            .map(|prepared| prepared.world_bounds());
                        if let (Some(slice_mode), Some(bounds)) = (kind.slice_mode(), bounds) {
                            let slice_state = active_slice_view_state(&mut session);
                            if slice_state.mode != slice_mode {
                                slice_state.set_mode(slice_mode);
                            }
                            slice_state.center_on_world(world, bounds);
                            render_quad_mpr_previews(&mut session)
                        } else {
                            Ok(())
                        }
                    } else {
                        Ok(())
                    }
                }
                QuadReferenceDrag::TranslateLine {
                    view,
                    start_crosshair_world,
                    start_pointer_world,
                    line_normal,
                    ..
                } => {
                    if view != kind {
                        return;
                    }
                    if let Some(pointer_world) = quad_mpr_world_point_from_viewport(
                        &session,
                        kind,
                        x,
                        y,
                        viewport_width,
                        viewport_height,
                    ) {
                        if let Some(slice_mode) = kind.slice_mode() {
                            let delta = (pointer_world - start_pointer_world).dot(line_normal);
                            let new_world = start_crosshair_world + line_normal * delta;
                            let slice_state = active_slice_view_state(&mut session);
                            if slice_state.mode != slice_mode {
                                slice_state.set_mode(slice_mode);
                            }
                            slice_state.set_crosshair_world(new_world);
                            render_quad_mpr_previews(&mut session)
                        } else {
                            Ok(())
                        }
                    } else {
                        Ok(())
                    }
                }
                QuadReferenceDrag::RotateLine {
                    view,
                    start_angle_rad,
                    start_orientation,
                    ..
                } => {
                    if view != kind {
                        return;
                    }
                    quad_mpr_angle_from_viewport(
                        &session,
                        kind,
                        x,
                        y,
                        viewport_width,
                        viewport_height,
                    )
                    .map(|current_angle_rad| {
                        let bounds = session
                            .prepared_volumes_by_series
                            .get(&session.active_series_uid)
                            .map(|prepared| prepared.world_bounds());
                        if let (Some(slice_mode), Some(bounds)) = (kind.slice_mode(), bounds) {
                            let slice_state = active_slice_view_state(&mut session);
                            if slice_state.mode != slice_mode {
                                slice_state.set_mode(slice_mode);
                            }
                            slice_state.orientation = start_orientation;
                            slice_state.rotate_about_normal(
                                normalized_angle_delta(current_angle_rad, start_angle_rad),
                                bounds,
                            );
                            render_quad_mpr_previews(&mut session)
                        } else {
                            Ok(())
                        }
                    })
                    .unwrap_or(Ok(()))
                }
            };
            if let Err(error) = result {
                info!("Failed to update quad MPR cursor interaction: {}", error);
            }
            return;
        }

        if session.drag_state.is_none() {
            let hovered =
                quad_reference_line_hit(&session, kind, x, y, viewport_width, viewport_height);
            if session.quad_reference_hover != hovered {
                session.quad_reference_hover = hovered;
                let _ = apply_viewport_state(&session);
            }
        }

        let Some(drag_state) = session.drag_state else {
            return;
        };
        if !matches!(session.active_tool, leaf_ui::ViewerTool::WindowLevel) {
            return;
        }
        let dx = x - drag_state.origin_x;
        let dy = y - drag_state.origin_y;
        let scalar_range = session
            .prepared_volumes_by_series
            .get(&session.active_series_uid)
            .map(|prepared| prepared.scalar_range());
        let result = if let Some((scalar_min, scalar_max)) = scalar_range {
            let sensitivity = (drag_state.start_window_width / 512.0).max(1.0);
            let center = drag_state.start_window_center + dx as f64 * sensitivity;
            let width = (drag_state.start_window_width - dy as f64 * sensitivity).max(1.0);
            active_slice_view_state(&mut session)
                .set_transfer_window(center, width, scalar_min, scalar_max);
            render_quad_mpr_previews(&mut session)
        } else {
            Ok(())
        };
        if let Err(error) = result {
            info!("Failed to update quad MPR interaction: {}", error);
        }
    });

    let session_for_quad_scroll = session.clone();
    viewer.on_quad_viewport_scroll(move |index, delta| {
        let Some(kind) = QuadViewportKind::from_index(index) else {
            return;
        };
        let mut session = session_for_quad_scroll.borrow_mut();
        set_selected_quad_viewport(&mut session, kind);
        if !session.volume_preview_active || !session.quad_viewport_active {
            let _ = apply_viewport_state(&session);
            return;
        }

        let result = if kind.is_dvr() {
            let zoom_factor = if delta > 0.0 {
                1.1
            } else if delta < 0.0 {
                0.9
            } else {
                1.0
            };
            active_volume_view_state(&mut session).zoom_by(zoom_factor);
            render_quad_single_preview(&mut session, QuadViewportKind::Dvr, false)
        } else {
            adjust_quad_crosshair_by_scroll(&mut session, kind, delta)
                .and_then(|_| render_quad_mpr_previews(&mut session))
        };
        if let Err(error) = result {
            info!("Failed to update quad viewport scroll: {}", error);
        }
    });

    let session_for_resize = session.clone();
    viewer.on_viewport_resized(move |width, height| {
        let mut session = session_for_resize.borrow_mut();
        let had_size = session.viewport_width > 0.0 && session.viewport_height > 0.0;
        update_viewport_dimensions(&mut session, width, height);
        // Update overlays if this is the first time we have valid dimensions
        if !had_size && width > 0.0 && height > 0.0 {
            let _ = update_measurement_overlays(&session);
        }
    });

    let session_for_reset = session.clone();
    viewer.on_reset_view(move || {
        let mut session = session_for_reset.borrow_mut();
        let result = if session.volume_preview_active {
            if session.quad_viewport_active {
                let focused_slice_mode = session.focused_quad_viewport.slice_mode();
                active_volume_view_state(&mut session).reset();
                let slice_state = active_slice_view_state(&mut session);
                slice_state.reset();
                slice_state.crosshair_world = None;
                if let Some(slice_mode) = focused_slice_mode {
                    slice_state.set_mode(slice_mode);
                }
            } else if session.advanced_preview_mode.is_dvr() {
                active_volume_view_state(&mut session).reset();
            } else {
                let slice_mode = session
                    .advanced_preview_mode
                    .slice_mode()
                    .unwrap_or_default();
                let slice_state = active_slice_view_state(&mut session);
                slice_state.reset();
                slice_state.crosshair_world = None;
                slice_state.set_mode(slice_mode);
            }
            render_or_show_volume_preview(&mut session)
        } else {
            session.active_frame_index = 0;
            session.selected_measurement_id = None;
            session.draft_measurement = None;
            session.handle_drag = None;
            reset_viewport_state(&mut session, false);
            update_viewer_image(&mut session).and_then(|_| update_measurements_model(&session))
        };
        if let Err(error) = result {
            info!("Failed to reset view: {}", error);
        }
    });

    let session_for_measurement_toggle = session.clone();
    viewer.on_toggle_measurements(move || {
        let mut session = session_for_measurement_toggle.borrow_mut();
        session.measurement_panel_visible = !session.measurement_panel_visible;
        if let Some(viewer) = session.viewer.upgrade() {
            viewer.set_measurement_panel_visible(session.measurement_panel_visible);
        }
    });

    let session_for_measurement_click = session.clone();
    viewer.on_measurement_clicked(move |measurement_id| {
        let mut session = session_for_measurement_click.borrow_mut();
        session.selected_measurement_id = Some(measurement_id.to_string());

        let selected = session
            .measurements_by_series
            .get(&session.active_series_uid)
            .and_then(|measurements| {
                measurements
                    .iter()
                    .find(|measurement| measurement.id == measurement_id.as_str())
            })
            .cloned();

        let Some(selected) = selected else {
            if let Err(error) = update_measurement_overlays(&session) {
                info!("Failed to update measurement selection: {}", error);
            }
            return;
        };

        let max_index = session
            .frames_by_series
            .get(&session.active_series_uid)
            .map(|frames| frames.len().saturating_sub(1))
            .unwrap_or(0);
        session.active_frame_index = selected.slice_index.min(max_index);
        let value_text = measurement_value_text(&selected, active_pixel_spacing(&session));

        let result = update_viewer_image(&mut session)
            .and_then(|_| update_measurements_model(&session))
            .and_then(|_| update_measurement_overlays(&session));
        if let Err(error) = result {
            info!("Failed to select measurement: {}", error);
            return;
        }

        if let Some(viewer) = session.viewer.upgrade() {
            viewer.set_connection_status(
                format!(
                    "Selected {} {}",
                    measurement_kind_label(&selected),
                    value_text
                )
                .into(),
            );
        }
    });

    let session_for_delete = session.clone();
    viewer.on_measurement_deleted(move |measurement_id| {
        let mut session = session_for_delete.borrow_mut();
        let measurement_id_str = measurement_id.to_string();
        let active_uid = session.active_series_uid.clone();

        // Remove from in-memory store
        if let Some(measurements) = session.measurements_by_series.get_mut(&active_uid) {
            measurements.retain(|m| m.id != measurement_id_str);
        }

        // Clear selection if it was selected
        if session.selected_measurement_id.as_deref() == Some(&measurement_id_str) {
            session.selected_measurement_id = None;
        }

        // Persist to DB
        persist_measurements(&session);

        // Update UI
        let _ = update_measurements_model(&session);
        let _ = update_measurement_overlays(&session);
    });

    let session_for_layout = session.clone();
    viewer.on_toggle_layout(move || {
        let mut session = session_for_layout.borrow_mut();
        session.quad_viewport_active = !session.quad_viewport_active;
        session.quad_reference_hover = None;
        session.quad_reference_drag = None;
        let result = if session.volume_preview_active {
            render_or_show_volume_preview(&mut session)
        } else if let Some(viewer) = session.viewer.upgrade() {
            viewer.set_layout_label(layout_label(&session).into());
            viewer.set_quad_view_active(false);
            viewer.set_focused_quad_viewport(session.focused_quad_viewport.index());
            Ok(())
        } else {
            Ok(())
        };
        if let Err(error) = result {
            info!("Failed to toggle viewport layout: {}", error);
        }
    });

    let session_for_volume_toggle = session.clone();
    let viewer_weak_for_close = viewer.as_weak();
    let imagebox_for_close = imagebox.clone();
    let session_for_advanced_mode = session.clone();
    viewer.on_cycle_advanced_preview_mode(move || {
        let mut session = session_for_advanced_mode.borrow_mut();
        session.advanced_preview_mode = session.advanced_preview_mode.next();
        session.focused_quad_viewport = session.advanced_preview_mode.quad_viewport();
        if let Some(slice_mode) = session.advanced_preview_mode.slice_mode() {
            let bounds = session
                .prepared_volumes_by_series
                .get(&session.active_series_uid)
                .map(|prepared| prepared.world_bounds());
            let slice_state = active_slice_view_state(&mut session);
            slice_state.set_mode(slice_mode);
            if let Some(bounds) = bounds {
                slice_state.center_on_crosshair(bounds);
            }
        }
        let result = if session.volume_preview_active {
            if session.quad_viewport_active {
                apply_viewport_state(&session)
            } else {
                render_or_show_volume_preview(&mut session)
            }
        } else if let Some(viewer) = session.viewer.upgrade() {
            viewer.set_advanced_preview_label(session.advanced_preview_mode.label().into());
            viewer.set_focused_quad_viewport(session.focused_quad_viewport.index());
            viewer.set_volume_mode_label(
                if session.advanced_preview_mode.is_dvr() {
                    volume_blend_mode_label(
                        session
                            .volume_view_state_by_series
                            .get(&session.active_series_uid)
                            .map(|state| state.blend_mode)
                            .unwrap_or_default(),
                    )
                } else {
                    slice_projection_mode_label(
                        session
                            .slice_view_state_by_series
                            .get(&session.active_series_uid)
                            .map(|state| state.projection_mode)
                            .unwrap_or_default(),
                    )
                }
                .into(),
            );
            Ok(())
        } else {
            Ok(())
        };
        if let Err(error) = result {
            info!("Failed to change advanced preview mode: {}", error);
        }
    });
    let session_for_volume_mode = session.clone();
    viewer.on_cycle_volume_mode(move || {
        let mut session = session_for_volume_mode.borrow_mut();
        let result = if session.advanced_preview_mode.is_dvr() {
            let next_mode = {
                let view_state = active_volume_view_state(&mut session);
                let next_mode = next_volume_blend_mode(view_state.blend_mode);
                view_state.blend_mode = next_mode;
                next_mode
            };
            if session.volume_preview_active {
                render_or_show_volume_preview(&mut session)
            } else {
                if let Some(viewer) = session.viewer.upgrade() {
                    viewer.set_volume_mode_label(volume_blend_mode_label(next_mode).into());
                }
                Ok(())
            }
        } else {
            let slice_mode = session
                .advanced_preview_mode
                .slice_mode()
                .unwrap_or_default();
            let default_half_thickness = session
                .prepared_volumes_by_series
                .get(&session.active_series_uid)
                .map(|prepared| prepared.slice_scroll_step(slice_mode) * 8.0)
                .unwrap_or(4.0);
            let next_mode = {
                let view_state = active_slice_view_state(&mut session);
                if view_state.mode != slice_mode {
                    view_state.set_mode(slice_mode);
                }
                view_state.cycle_projection_mode(default_half_thickness);
                view_state.projection_mode
            };
            if session.volume_preview_active {
                render_or_show_volume_preview(&mut session)
            } else if let Some(viewer) = session.viewer.upgrade() {
                viewer.set_volume_mode_label(slice_projection_mode_label(next_mode).into());
                Ok(())
            } else {
                Ok(())
            }
        };
        if let Err(error) = result {
            info!("Failed to change volume blend mode: {}", error);
        }
    });
    viewer.on_toggle_volume_preview(move || {
        let mut session = session_for_volume_toggle.borrow_mut();
        let result = if session.volume_preview_active {
            session.volume_preview_active = false;
            session.volume_drag_state = None;
            session.quad_reference_hover = None;
            session.quad_reference_drag = None;
            update_viewer_image(&mut session).and_then(|_| update_measurements_model(&session))
        } else {
            session.viewport_scale = 1.0;
            session.viewport_offset_x = 0.0;
            session.viewport_offset_y = 0.0;
            render_or_show_volume_preview(&mut session)
        };
        if let Err(error) = result {
            session.volume_preview_active = false;
            if let Some(viewer) = session.viewer.upgrade() {
                viewer.set_volume_preview_active(false);
                viewer.set_connection_status(format!("3D preview failed: {error}").into());
            }
        }
    });
    viewer.window().on_close_requested(move || {
        if let Some(viewer) = viewer_weak_for_close.upgrade() {
            if let Some(geometry) = capture_window_geometry(viewer.window()) {
                if let Err(error) =
                    save_window_geometry(&imagebox_for_close, VIEWER_WINDOW_GEOMETRY_KEY, &geometry)
                {
                    info!("Failed to save viewer window geometry: {}", error);
                }
            }
        }
        slint::CloseRequestResponse::HideWindow
    });

    viewer
        .show()
        .map_err(|error| LeafError::Render(error.to_string()))?;
    Ok(viewer)
}

fn update_series_model(session: &ViewerSession) -> LeafResult<()> {
    let entries = session
        .series
        .iter()
        .map(|series| {
            let uid = &series.series_uid.0;
            let (thumbnail, has_thumbnail) = match session.thumbnails_by_series.get(uid) {
                Some(img) => (img.clone(), true),
                None => (slint::Image::default(), false),
            };
            leaf_ui::SeriesThumbnail {
                series_uid: uid.clone().into(),
                series_number: series.series_number.unwrap_or_default(),
                modality: series.modality.clone().into(),
                description: series
                    .series_description
                    .clone()
                    .unwrap_or_else(|| "-".to_string())
                    .into(),
                instance_count: session
                    .instances_by_series
                    .get(uid)
                    .map(|instances| instances.len() as i32)
                    .unwrap_or(0),
                active: *uid == session.active_series_uid,
                thumbnail,
                has_thumbnail,
            }
        })
        .collect::<Vec<_>>();
    session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?
        .set_series_list(ModelRc::from(Rc::new(VecModel::from(entries))));
    Ok(())
}

fn update_measurements_model(session: &ViewerSession) -> LeafResult<()> {
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    if session.volume_preview_active {
        viewer.set_measurements(empty_measurement_model());
        return Ok(());
    }
    let pixel_spacing = active_pixel_spacing(session);
    let entries = session
        .measurements_by_series
        .get(&session.active_series_uid)
        .map(|measurements| {
            measurements
                .iter()
                .map(|measurement| measurement_entry(measurement, pixel_spacing))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    viewer.set_measurements(ModelRc::from(Rc::new(VecModel::from(entries))));
    Ok(())
}

fn active_volume_view_state(session: &mut ViewerSession) -> &mut VolumeViewState {
    session
        .volume_view_state_by_series
        .entry(session.active_series_uid.clone())
        .or_default()
}

fn active_slice_view_state(session: &mut ViewerSession) -> &mut SlicePreviewState {
    session
        .slice_view_state_by_series
        .entry(session.active_series_uid.clone())
        .or_default()
}

fn layout_label(session: &ViewerSession) -> &'static str {
    if session.quad_viewport_active {
        "Quad"
    } else {
        "1Up"
    }
}

fn set_selected_quad_viewport(session: &mut ViewerSession, kind: QuadViewportKind) {
    session.focused_quad_viewport = kind;
    session.advanced_preview_mode = kind.advanced_preview_mode();
}

fn quad_preview(
    session: &ViewerSession,
    kind: QuadViewportKind,
) -> Option<&AdvancedViewportPreview> {
    session.quad_previews_by_kind.get(&kind)
}

fn quad_preview_info(session: &ViewerSession, kind: QuadViewportKind) -> String {
    quad_preview(session, kind)
        .map(|preview| preview.info.clone())
        .unwrap_or_default()
}

fn quad_tile_max_dimensions(session: &ViewerSession) -> (f32, f32) {
    let width = if session.viewport_width > 0.0 {
        session.viewport_width * 0.5
    } else {
        512.0
    };
    let height = if session.viewport_height > 0.0 {
        session.viewport_height * 0.5
    } else {
        512.0
    };
    (width.max(256.0), height.max(256.0))
}

fn quad_mpr_preview_info(kind: QuadViewportKind, state: SlicePreviewState) -> String {
    format!(
        "{} {}",
        kind.title(),
        slice_projection_mode_label(state.projection_mode)
    )
}

fn quad_viewport_geometry(
    session: &ViewerSession,
    kind: QuadViewportKind,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<ViewportGeometry> {
    let preview = quad_preview(session, kind)?;
    displayed_image_geometry(
        viewport_width,
        viewport_height,
        preview.width,
        preview.height,
        1.0,
        0.0,
        0.0,
    )
}

fn quad_world_to_viewport_point(
    plane: &SlicePlane,
    geometry: ViewportGeometry,
    world: glam::DVec3,
) -> (f32, f32) {
    let (uv, _) = plane.world_to_point(world);
    (
        geometry.image_origin_x + uv.x as f32 * geometry.image_width,
        geometry.image_origin_y + uv.y as f32 * geometry.image_height,
    )
}

fn clip_line_to_geometry(
    point_x: f32,
    point_y: f32,
    dir_x: f32,
    dir_y: f32,
    geometry: ViewportGeometry,
) -> Option<((f32, f32), (f32, f32))> {
    let min_x = geometry.image_origin_x;
    let min_y = geometry.image_origin_y;
    let max_x = geometry.image_origin_x + geometry.image_width;
    let max_y = geometry.image_origin_y + geometry.image_height;
    let mut intersections = Vec::new();
    const EPSILON: f32 = 1.0e-4;

    if dir_x.abs() > EPSILON {
        for edge_x in [min_x, max_x] {
            let t = (edge_x - point_x) / dir_x;
            let y = point_y + t * dir_y;
            if y >= min_y - EPSILON && y <= max_y + EPSILON {
                intersections.push((t, edge_x, y.clamp(min_y, max_y)));
            }
        }
    }
    if dir_y.abs() > EPSILON {
        for edge_y in [min_y, max_y] {
            let t = (edge_y - point_y) / dir_y;
            let x = point_x + t * dir_x;
            if x >= min_x - EPSILON && x <= max_x + EPSILON {
                intersections.push((t, x.clamp(min_x, max_x), edge_y));
            }
        }
    }

    intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    intersections.dedup_by(|a, b| (a.1 - b.1).abs() <= 0.5 && (a.2 - b.2).abs() <= 0.5);
    let start = intersections.first()?;
    let end = intersections.last()?;
    ((start.1 - end.1).abs() > EPSILON || (start.2 - end.2).abs() > EPSILON)
        .then_some(((start.1, start.2), (end.1, end.2)))
}

fn quad_reference_line_for_plane(
    current_plane: &SlicePlane,
    other_plane: &SlicePlane,
    source_kind: QuadViewportKind,
    shared_world: glam::DVec3,
    geometry: ViewportGeometry,
) -> Option<QuadReferenceLineOverlay> {
    let direction = current_plane.normal().cross(other_plane.normal());
    if direction.length_squared() <= 1.0e-10 {
        return None;
    }
    let line_extent = current_plane.width.max(current_plane.height).max(1.0);
    let dir = direction.normalize();
    let (center_x, center_y) = quad_world_to_viewport_point(current_plane, geometry, shared_world);
    let (dir_x, dir_y) =
        quad_world_to_viewport_point(current_plane, geometry, shared_world + dir * line_extent);
    let ((start_x, start_y), (end_x, end_y)) = clip_line_to_geometry(
        center_x,
        center_y,
        dir_x - center_x,
        dir_y - center_y,
        geometry,
    )?;
    Some(QuadReferenceLineOverlay {
        commands: format!("M {start_x:.1} {start_y:.1} L {end_x:.1} {end_y:.1}"),
        start_x,
        start_y,
        end_x,
        end_y,
        handle1_x: (center_x + start_x) * 0.5,
        handle1_y: (center_y + start_y) * 0.5,
        handle2_x: (center_x + end_x) * 0.5,
        handle2_y: (center_y + end_y) * 0.5,
        source_kind,
    })
}

fn rebuild_quad_reference_lines(session: &mut ViewerSession) -> LeafResult<()> {
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    if !session.volume_preview_active || !session.quad_viewport_active {
        viewer.set_quad_axial_reference_lines(empty_quad_reference_line_model());
        viewer.set_quad_coronal_reference_lines(empty_quad_reference_line_model());
        viewer.set_quad_sagittal_reference_lines(empty_quad_reference_line_model());
        session.quad_reference_lines_by_kind.clear();
        return Ok(());
    }

    let prepared = session
        .prepared_volumes_by_series
        .get(&session.active_series_uid)
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    let (tile_width, tile_height) = quad_tile_max_dimensions(session);
    for kind in [
        QuadViewportKind::Axial,
        QuadViewportKind::Coronal,
        QuadViewportKind::Sagittal,
    ] {
        let Some(current_state) = quad_slice_view_state_for_kind(session, prepared, kind) else {
            continue;
        };
        let Some(geometry) = quad_viewport_geometry(session, kind, tile_width, tile_height) else {
            continue;
        };
        let current_plane = current_state.slice_plane(prepared.world_bounds());
        let shared_world = current_state.crosshair_world(prepared.world_bounds());
        let overlays = kind
            .linked_mpr_views()
            .into_iter()
            .filter_map(|other_kind| {
                quad_slice_view_state_for_kind(session, prepared, other_kind).and_then(
                    |other_state| {
                        quad_reference_line_for_plane(
                            &current_plane,
                            &other_state.slice_plane(prepared.world_bounds()),
                            other_kind,
                            shared_world,
                            geometry,
                        )
                    },
                )
            })
            .collect::<Vec<_>>();
        session.quad_reference_lines_by_kind.insert(kind, overlays);
        let model = quad_reference_line_model(session, kind);
        match kind {
            QuadViewportKind::Axial => viewer.set_quad_axial_reference_lines(model),
            QuadViewportKind::Coronal => viewer.set_quad_coronal_reference_lines(model),
            QuadViewportKind::Sagittal => viewer.set_quad_sagittal_reference_lines(model),
            QuadViewportKind::Dvr => {}
        }
    }
    Ok(())
}

fn quad_mpr_angle_from_viewport(
    session: &ViewerSession,
    kind: QuadViewportKind,
    x: f32,
    y: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<f64> {
    let prepared = session
        .prepared_volumes_by_series
        .get(&session.active_series_uid)?;
    let slice_state = quad_slice_view_state_for_kind(session, prepared, kind)?;
    let bounds = prepared.world_bounds();
    let plane = slice_state.slice_plane(bounds);
    let shared_world = slice_state.crosshair_world(bounds);
    let pointer_world =
        quad_mpr_world_point_from_viewport(session, kind, x, y, viewport_width, viewport_height)?;
    let offset = pointer_world - shared_world;
    let dx = offset.dot(plane.right);
    let dy = offset.dot(plane.up);
    if dx.abs() < 1.0e-4_f64 && dy.abs() < 1.0e-4_f64 {
        return None;
    }
    Some(dy.atan2(dx))
}

fn quad_crosshair_viewport_point(
    session: &ViewerSession,
    kind: QuadViewportKind,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<(f32, f32)> {
    let geometry = quad_viewport_geometry(session, kind, viewport_width, viewport_height)?;
    let prepared = session
        .prepared_volumes_by_series
        .get(&session.active_series_uid)?;
    let slice_state = quad_slice_view_state_for_kind(session, prepared, kind)?;
    let bounds = prepared.world_bounds();
    let plane = slice_state.slice_plane(bounds);
    let shared_world = slice_state.crosshair_world(bounds);
    Some(quad_world_to_viewport_point(&plane, geometry, shared_world))
}

fn normalized_angle_delta(current_angle_rad: f64, start_angle_rad: f64) -> f64 {
    let mut delta = current_angle_rad - start_angle_rad;
    while delta <= -std::f64::consts::PI {
        delta += std::f64::consts::TAU;
    }
    while delta > std::f64::consts::PI {
        delta -= std::f64::consts::TAU;
    }
    delta
}

fn point_to_segment_distance_sq(
    point_x: f32,
    point_y: f32,
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
) -> f32 {
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    let length_sq = dx * dx + dy * dy;
    if length_sq <= 1.0e-4 {
        let px = point_x - start_x;
        let py = point_y - start_y;
        return px * px + py * py;
    }
    let t = (((point_x - start_x) * dx) + ((point_y - start_y) * dy)) / length_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = start_x + dx * t;
    let proj_y = start_y + dy * t;
    let px = point_x - proj_x;
    let py = point_y - proj_y;
    px * px + py * py
}

fn quad_reference_line_hit(
    session: &ViewerSession,
    kind: QuadViewportKind,
    x: f32,
    y: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<QuadReferenceSelection> {
    const CENTER_THRESHOLD_SQ: f32 = 144.0;
    const HANDLE_THRESHOLD_SQ: f32 = 100.0;
    const LINE_THRESHOLD_SQ: f32 = 81.0;

    if let Some((center_x, center_y)) =
        quad_crosshair_viewport_point(session, kind, viewport_width, viewport_height)
    {
        let dx = x - center_x;
        let dy = y - center_y;
        if dx * dx + dy * dy <= CENTER_THRESHOLD_SQ {
            return Some(QuadReferenceSelection {
                view: kind,
                target: QuadReferenceTarget::Center,
            });
        }
    }

    if let Some((_, source_kind)) = session
        .quad_reference_lines_by_kind
        .get(&kind)
        .into_iter()
        .flat_map(|lines| lines.iter())
        .filter_map(|line| {
            let distance_sq = point_to_segment_distance_sq(
                x,
                y,
                line.handle1_x,
                line.handle1_y,
                line.handle1_x,
                line.handle1_y,
            )
            .min(point_to_segment_distance_sq(
                x,
                y,
                line.handle2_x,
                line.handle2_y,
                line.handle2_x,
                line.handle2_y,
            ));
            (distance_sq <= HANDLE_THRESHOLD_SQ).then_some((distance_sq, line.source_kind))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
    {
        return Some(QuadReferenceSelection {
            view: kind,
            target: QuadReferenceTarget::RotateLine(source_kind),
        });
    }

    session
        .quad_reference_lines_by_kind
        .get(&kind)
        .into_iter()
        .flat_map(|lines| lines.iter())
        .filter_map(|line| {
            let line_distance_sq = point_to_segment_distance_sq(
                x,
                y,
                line.start_x,
                line.start_y,
                line.end_x,
                line.end_y,
            );
            let handle_distance_sq = point_to_segment_distance_sq(
                x,
                y,
                line.handle1_x,
                line.handle1_y,
                line.handle1_x,
                line.handle1_y,
            )
            .min(point_to_segment_distance_sq(
                x,
                y,
                line.handle2_x,
                line.handle2_y,
                line.handle2_x,
                line.handle2_y,
            ));
            (line_distance_sq <= LINE_THRESHOLD_SQ && handle_distance_sq > HANDLE_THRESHOLD_SQ)
                .then_some((
                    line_distance_sq,
                    QuadReferenceSelection {
                        view: kind,
                        target: QuadReferenceTarget::TranslateLine(line.source_kind),
                    },
                ))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        .map(|(_, selection)| selection)
}

fn quad_reference_line_highlighted(
    session: &ViewerSession,
    view: QuadViewportKind,
    source_kind: QuadViewportKind,
) -> bool {
    let drag_active = matches!(
        session.quad_reference_drag,
        Some(QuadReferenceDrag::TranslateLine {
            view: drag_view,
            line_kind,
            ..
        } | QuadReferenceDrag::RotateLine {
            view: drag_view,
            line_kind,
            ..
        }) if (drag_view, line_kind) == (view, source_kind)
    );
    let hover_active = matches!(
        session.quad_reference_hover,
        Some(QuadReferenceSelection {
            view: hover_view,
            target: QuadReferenceTarget::TranslateLine(line_kind)
                | QuadReferenceTarget::RotateLine(line_kind),
        }) if (hover_view, line_kind) == (view, source_kind)
    );
    drag_active || hover_active
}

fn quad_reference_center_highlighted(session: &ViewerSession, view: QuadViewportKind) -> bool {
    matches!(
        session.quad_reference_drag,
        Some(QuadReferenceDrag::Center { view: drag_view }) if drag_view == view
    ) || matches!(
        session.quad_reference_hover,
        Some(QuadReferenceSelection {
            view: hover_view,
            target: QuadReferenceTarget::Center,
        }) if hover_view == view
    )
}

fn volume_blend_mode_label(mode: VolumeBlendMode) -> &'static str {
    match mode {
        VolumeBlendMode::Composite => "Comp",
        VolumeBlendMode::MaximumIntensity => "MIP",
        VolumeBlendMode::MinimumIntensity => "MinIP",
        VolumeBlendMode::AverageIntensity => "Avg",
    }
}

fn next_volume_blend_mode(mode: VolumeBlendMode) -> VolumeBlendMode {
    match mode {
        VolumeBlendMode::Composite => VolumeBlendMode::MaximumIntensity,
        VolumeBlendMode::MaximumIntensity => VolumeBlendMode::MinimumIntensity,
        VolumeBlendMode::MinimumIntensity => VolumeBlendMode::AverageIntensity,
        VolumeBlendMode::AverageIntensity => VolumeBlendMode::Composite,
    }
}

fn slice_projection_mode_label(mode: SliceProjectionMode) -> &'static str {
    match mode {
        SliceProjectionMode::Thin => "Thin",
        SliceProjectionMode::MaximumIntensity => "MIP",
        SliceProjectionMode::MinimumIntensity => "MinIP",
        SliceProjectionMode::AverageIntensity => "Avg",
    }
}

fn apply_volume_drag(
    session: &mut ViewerSession,
    start_view_state: VolumeViewState,
    delta_x: f32,
    delta_y: f32,
    button: i32,
) {
    let active_tool = session.active_tool;
    let scalar_range = session
        .prepared_volumes_by_series
        .get(&session.active_series_uid)
        .map(|prepared| prepared.scalar_range());
    let view_state = active_volume_view_state(session);
    *view_state = start_view_state;
    if let Some((scalar_min, scalar_max)) = scalar_range {
        view_state.ensure_transfer_window(scalar_min, scalar_max);
    }

    // Right-click always orbits, regardless of active tool.
    if button == 2 {
        view_state.orbit(delta_x as f64 * 0.6, -delta_y as f64 * 0.6);
        return;
    }

    match active_tool {
        leaf_ui::ViewerTool::WindowLevel => {
            if let Some((scalar_min, scalar_max)) = scalar_range {
                let (start_center, start_width) =
                    start_view_state.transfer_window(scalar_min, scalar_max);
                let range = (scalar_max - scalar_min).max(1.0);
                let sensitivity = (range / 450.0).max(1.0);
                view_state.set_transfer_window(
                    start_center + delta_y as f64 * sensitivity,
                    start_width + delta_x as f64 * sensitivity,
                    scalar_min,
                    scalar_max,
                );
            }
        }
        leaf_ui::ViewerTool::Pan => view_state.pan(-delta_x as f64, -delta_y as f64),
        leaf_ui::ViewerTool::Zoom => {
            let factor = (1.0 - delta_y as f64 * 0.004).clamp(0.1, 10.0);
            view_state.zoom_by(factor);
        }
        _ => view_state.orbit(delta_x as f64 * 0.6, -delta_y as f64 * 0.6),
    }
}

fn empty_measurement_model() -> ModelRc<leaf_ui::MeasurementEntry> {
    ModelRc::from(Rc::new(VecModel::from(
        Vec::<leaf_ui::MeasurementEntry>::new(),
    )))
}

fn empty_measurement_overlay_model() -> ModelRc<leaf_ui::MeasurementOverlay> {
    ModelRc::from(Rc::new(VecModel::from(
        Vec::<leaf_ui::MeasurementOverlay>::new(),
    )))
}

fn empty_quad_reference_line_model() -> ModelRc<leaf_ui::QuadReferenceLine> {
    ModelRc::from(Rc::new(VecModel::from(
        Vec::<leaf_ui::QuadReferenceLine>::new(),
    )))
}

fn quad_reference_line_model(
    session: &ViewerSession,
    kind: QuadViewportKind,
) -> ModelRc<leaf_ui::QuadReferenceLine> {
    let entries = session
        .quad_reference_lines_by_kind
        .get(&kind)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|overlay| leaf_ui::QuadReferenceLine {
            commands: overlay.commands.into(),
            h1_x: overlay.handle1_x,
            h1_y: overlay.handle1_y,
            h2_x: overlay.handle2_x,
            h2_y: overlay.handle2_y,
            kind: overlay.source_kind.index(),
            active: quad_reference_line_highlighted(session, kind, overlay.source_kind),
        })
        .collect::<Vec<_>>();
    ModelRc::from(Rc::new(VecModel::from(entries)))
}

fn update_viewport_dimensions(
    session: &mut ViewerSession,
    viewport_width: f32,
    viewport_height: f32,
) {
    if viewport_width > 0.0 {
        session.viewport_width = viewport_width;
    }
    if viewport_height > 0.0 {
        session.viewport_height = viewport_height;
    }
}

fn measurement_entry(
    measurement: &Measurement,
    pixel_spacing: (f64, f64),
) -> leaf_ui::MeasurementEntry {
    leaf_ui::MeasurementEntry {
        id: measurement.id.clone().into(),
        kind: measurement_kind_label(measurement).into(),
        value: measurement_value_text(measurement, pixel_spacing).into(),
        label: measurement.label.clone().unwrap_or_default().into(),
        slice_index: (measurement.slice_index + 1) as i32,
    }
}

fn measurement_kind_label(measurement: &Measurement) -> &'static str {
    match &measurement.kind {
        MeasurementKind::Line { .. } => "Line",
        MeasurementKind::Angle { .. } => "Angle",
        MeasurementKind::RectangleRoi { .. } => "ROI",
        MeasurementKind::EllipseRoi { .. } => "Ellipse ROI",
        MeasurementKind::PolygonRoi { .. } => "Polygon ROI",
        MeasurementKind::PixelProbe { .. } => "Probe",
    }
}

fn measurement_value_text(measurement: &Measurement, pixel_spacing: (f64, f64)) -> String {
    match measurement.compute(pixel_spacing).value {
        MeasurementValue::Distance { mm } => format_distance_mm(mm),
        MeasurementValue::Angle { degrees } => format!("{degrees:.1}\u{b0}"),
        MeasurementValue::RoiStats { area_mm2, .. } => format!("{area_mm2:.1} mm\u{b2}"),
        MeasurementValue::PixelValue { value, unit } => format!("{value:.0} {unit}"),
    }
}

fn format_distance_mm(mm: f64) -> String {
    if mm >= 100.0 {
        format!("{mm:.0} mm")
    } else if mm >= 10.0 {
        format!("{mm:.1} mm")
    } else {
        format!("{mm:.2} mm")
    }
}

fn active_pixel_spacing(session: &ViewerSession) -> (f64, f64) {
    session
        .series
        .iter()
        .find(|series| series.series_uid.0 == session.active_series_uid)
        .and_then(|series| series.pixel_spacing)
        .unwrap_or((1.0, 1.0))
}

fn update_measurement_overlays(session: &ViewerSession) -> LeafResult<()> {
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    if session.volume_preview_active {
        viewer.set_measurement_overlays(empty_measurement_overlay_model());
        return Ok(());
    }

    let has_visible = session
        .measurements_by_series
        .get(&session.active_series_uid)
        .map(|m| {
            m.iter()
                .any(|m| m.slice_index == session.active_frame_index)
        })
        .unwrap_or(false);
    if !has_visible && session.draft_measurement.is_none() {
        viewer.set_measurement_overlays(empty_measurement_overlay_model());
        return Ok(());
    }

    let pixel_spacing = active_pixel_spacing(session);

    let mut overlays = session
        .measurements_by_series
        .get(&session.active_series_uid)
        .into_iter()
        .flat_map(|measurements| measurements.iter())
        .filter(|measurement| measurement.slice_index == session.active_frame_index)
        .filter_map(|measurement| {
            measurement_to_overlay(
                session,
                measurement,
                pixel_spacing,
                session.selected_measurement_id.as_deref() == Some(measurement.id.as_str()),
                false,
            )
        })
        .collect::<Vec<_>>();

    if let Some(draft) = session.draft_measurement.as_ref() {
        let temp = match draft {
            DraftMeasurement::Line { start, end } => Measurement::line("", 0, *start, *end),
            DraftMeasurement::Angle { vertex, arm1, arm2 } => {
                Measurement::angle("", 0, *vertex, *arm1, arm2.unwrap_or(*arm1))
            }
            DraftMeasurement::Rectangle { corner1, corner2 } => {
                Measurement::rectangle_roi("", 0, *corner1, *corner2)
            }
            DraftMeasurement::Ellipse { center, corner } => {
                let rx = (corner.x - center.x).abs().max(0.1);
                let ry = (corner.y - center.y).abs().max(0.1);
                Measurement::ellipse_roi("", 0, *center, rx, ry)
            }
        };
        if let Some(overlay) = measurement_to_overlay(session, &temp, pixel_spacing, false, true) {
            overlays.push(overlay);
        }
    }

    viewer.set_measurement_overlays(ModelRc::from(Rc::new(VecModel::from(overlays))));
    Ok(())
}

fn measurement_to_overlay(
    session: &ViewerSession,
    measurement: &Measurement,
    pixel_spacing: (f64, f64),
    selected: bool,
    draft: bool,
) -> Option<leaf_ui::MeasurementOverlay> {
    let label = measurement_value_text(measurement, pixel_spacing);
    match &measurement.kind {
        MeasurementKind::Line { start, end } => {
            let (sx, sy) = image_to_viewport_point(session, *start)?;
            let (ex, ey) = image_to_viewport_point(session, *end)?;
            let dx = ex - sx;
            let dy = ey - sy;
            if (dx * dx + dy * dy).sqrt() < 1.0 {
                return None;
            }
            Some(leaf_ui::MeasurementOverlay {
                id: measurement.id.clone().into(),
                commands: format!("M {sx:.1} {sy:.1} L {ex:.1} {ey:.1}").into(),
                label_x: ex + 6.0,
                label_y: (ey - 16.0).max(6.0),
                label: label.into(),
                selected,
                draft,
                handle_count: 2,
                h1_x: sx,
                h1_y: sy,
                h2_x: ex,
                h2_y: ey,
                h3_x: 0.0,
                h3_y: 0.0,
                h4_x: 0.0,
                h4_y: 0.0,
            })
        }
        MeasurementKind::Angle { vertex, arm1, arm2 } => {
            let (vx, vy) = image_to_viewport_point(session, *vertex)?;
            let (a1x, a1y) = image_to_viewport_point(session, *arm1)?;
            let (a2x, a2y) = image_to_viewport_point(session, *arm2)?;
            Some(leaf_ui::MeasurementOverlay {
                id: measurement.id.clone().into(),
                commands: format!("M {a1x:.1} {a1y:.1} L {vx:.1} {vy:.1} L {a2x:.1} {a2y:.1}")
                    .into(),
                label_x: vx + 6.0,
                label_y: (vy - 16.0).max(6.0),
                label: label.into(),
                selected,
                draft,
                handle_count: 3,
                h1_x: vx,
                h1_y: vy,
                h2_x: a1x,
                h2_y: a1y,
                h3_x: a2x,
                h3_y: a2y,
                h4_x: 0.0,
                h4_y: 0.0,
            })
        }
        MeasurementKind::RectangleRoi {
            top_left,
            bottom_right,
        } => {
            let (x1, y1) = image_to_viewport_point(session, *top_left)?;
            let (x2, y2) = image_to_viewport_point(session, *bottom_right)?;
            Some(leaf_ui::MeasurementOverlay {
                id: measurement.id.clone().into(),
                commands: format!(
                    "M {x1:.1} {y1:.1} L {x2:.1} {y1:.1} L {x2:.1} {y2:.1} L {x1:.1} {y2:.1} Z"
                )
                .into(),
                label_x: x2 + 6.0,
                label_y: (y1 - 16.0).max(6.0),
                label: label.into(),
                selected,
                draft,
                handle_count: 4,
                h1_x: x1,
                h1_y: y1,
                h2_x: x2,
                h2_y: y1,
                h3_x: x2,
                h3_y: y2,
                h4_x: x1,
                h4_y: y2,
            })
        }
        MeasurementKind::EllipseRoi {
            center,
            radius_x,
            radius_y,
        } => {
            let (cx, cy) = image_to_viewport_point(session, *center)?;
            let corner = DVec2::new(center.x + radius_x, center.y + radius_y);
            let (corner_x, corner_y) = image_to_viewport_point(session, corner)?;
            let rx_vp = (corner_x - cx).abs();
            let ry_vp = (corner_y - cy).abs();
            if rx_vp < 1.0 || ry_vp < 1.0 {
                return None;
            }
            let commands = format!(
                "M {:.1} {:.1} A {:.1} {:.1} 0 1 0 {:.1} {:.1} A {:.1} {:.1} 0 1 0 {:.1} {:.1}",
                cx - rx_vp,
                cy,
                rx_vp,
                ry_vp,
                cx + rx_vp,
                cy,
                rx_vp,
                ry_vp,
                cx - rx_vp,
                cy
            );
            Some(leaf_ui::MeasurementOverlay {
                id: measurement.id.clone().into(),
                commands: commands.into(),
                label_x: cx + rx_vp + 6.0,
                label_y: (cy - 16.0).max(6.0),
                label: label.into(),
                selected,
                draft,
                handle_count: 2,
                h1_x: cx,
                h1_y: cy,
                h2_x: cx + rx_vp,
                h2_y: cy + ry_vp,
                h3_x: 0.0,
                h3_y: 0.0,
                h4_x: 0.0,
                h4_y: 0.0,
            })
        }
        _ => None,
    }
}

fn find_handle_at(session: &ViewerSession, x: f32, y: f32) -> Option<(String, usize)> {
    let threshold_sq = 64.0f32; // 8px radius
    let measurements = session
        .measurements_by_series
        .get(&session.active_series_uid)?;
    for measurement in measurements
        .iter()
        .filter(|m| m.slice_index == session.active_frame_index)
    {
        let handles = measurement.handle_positions();
        for (idx, handle) in handles.iter().enumerate() {
            if let Some((hx, hy)) = image_to_viewport_point(session, *handle) {
                let dx = hx - x;
                let dy = hy - y;
                if dx * dx + dy * dy <= threshold_sq {
                    return Some((measurement.id.clone(), idx));
                }
            }
        }
    }
    None
}

fn finalize_measurement(session: &mut ViewerSession) -> LeafResult<()> {
    session.drag_state = None;

    let Some(draft) = session.draft_measurement.take() else {
        return update_measurement_overlays(session);
    };

    let measurement = match draft {
        DraftMeasurement::Line { start, end } => {
            if (end - start).length() < 2.0 {
                return update_measurement_overlays(session);
            }
            Measurement::line(
                &session.active_series_uid,
                session.active_frame_index,
                start,
                end,
            )
        }
        DraftMeasurement::Angle { vertex, arm1, arm2 } => {
            let arm2 = arm2.unwrap_or(arm1);
            if (arm1 - vertex).length() < 2.0 || (arm2 - vertex).length() < 2.0 {
                return update_measurement_overlays(session);
            }
            Measurement::angle(
                &session.active_series_uid,
                session.active_frame_index,
                vertex,
                arm1,
                arm2,
            )
        }
        DraftMeasurement::Rectangle { corner1, corner2 } => {
            if (corner2 - corner1).length() < 2.0 {
                return update_measurement_overlays(session);
            }
            Measurement::rectangle_roi(
                &session.active_series_uid,
                session.active_frame_index,
                corner1,
                corner2,
            )
        }
        DraftMeasurement::Ellipse { center, corner } => {
            let rx = (corner.x - center.x).abs();
            let ry = (corner.y - center.y).abs();
            if rx < 1.0 || ry < 1.0 {
                return update_measurement_overlays(session);
            }
            Measurement::ellipse_roi(
                &session.active_series_uid,
                session.active_frame_index,
                center,
                rx,
                ry,
            )
        }
    };

    let kind_label = measurement_kind_label(&measurement);
    let value_text = measurement_value_text(&measurement, active_pixel_spacing(session));
    let active_series_uid = session.active_series_uid.clone();
    session.selected_measurement_id = Some(measurement.id.clone());
    session
        .measurements_by_series
        .entry(active_series_uid)
        .or_default()
        .push(measurement);

    if !session.measurement_panel_visible {
        session.measurement_panel_visible = true;
        if let Some(viewer) = session.viewer.upgrade() {
            viewer.set_measurement_panel_visible(true);
        }
    }

    update_measurements_model(session)?;
    update_measurement_overlays(session)?;
    persist_measurements(session);

    if let Some(viewer) = session.viewer.upgrade() {
        viewer.set_connection_status(format!("Created {} {}", kind_label, value_text).into());
    }

    Ok(())
}

fn persist_measurements(session: &ViewerSession) {
    let series_uid = &session.active_series_uid;
    if let Some(measurements) = session.measurements_by_series.get(series_uid) {
        if let Ok(json) = serde_json::to_string(measurements) {
            if let Err(e) = session.imagebox.store_measurements(series_uid, &json) {
                info!("Failed to persist measurements: {}", e);
            }
        }
    } else {
        let _ = session.imagebox.delete_measurements(series_uid);
    }
}

fn lut_label(name: &str) -> &'static str {
    LUT_PRESETS
        .iter()
        .find_map(|(label, lut_name)| (*lut_name == name).then_some(*label))
        .unwrap_or("Gray")
}

fn next_lut_name(current: &str) -> &'static str {
    LUT_PRESETS
        .iter()
        .position(|(_, lut_name)| *lut_name == current)
        .map(|index| LUT_PRESETS[(index + 1) % LUT_PRESETS.len()].1)
        .unwrap_or(DEFAULT_LUT_NAME)
}

fn resolve_color_lut(name: &str) -> ColorLut {
    match name {
        "grayscale_inverted" => ColorLut::grayscale_inverted(),
        "hot_iron" => ColorLut::hot_iron(),
        "bone" => ColorLut::bone(),
        _ => ColorLut::grayscale(),
    }
}

fn apply_overlays_to_rgba(
    rgba: &mut [u8],
    image_width: u32,
    image_height: u32,
    overlays: &[OverlayBitmap],
) {
    let image_width = image_width as usize;
    let image_height = image_height as usize;
    for overlay in overlays {
        let origin_x = overlay.origin.1 as isize - 1;
        let origin_y = overlay.origin.0 as isize - 1;
        let columns = overlay.columns as usize;
        let rows = overlay.rows as usize;
        for row in 0..rows {
            for col in 0..columns {
                let bitmap_index = row * columns + col;
                if overlay
                    .bitmap
                    .get(bitmap_index)
                    .copied()
                    .unwrap_or_default()
                    == 0
                {
                    continue;
                }
                let target_x = origin_x + col as isize;
                let target_y = origin_y + row as isize;
                if target_x < 0
                    || target_y < 0
                    || target_x >= image_width as isize
                    || target_y >= image_height as isize
                {
                    continue;
                }
                let pixel_index =
                    (target_y as usize * image_width + target_x as usize) * OVERLAY_COLOR.len();
                rgba[pixel_index..pixel_index + OVERLAY_COLOR.len()]
                    .copy_from_slice(&OVERLAY_COLOR);
            }
        }
    }
}

fn frame_cache_key(session: &ViewerSession, frame_ref: &FrameRef) -> FrameCacheKey {
    let (has_window_override, window_center_bits, window_width_bits) =
        match session.window_center.zip(session.window_width) {
            Some((center, width)) => (true, center.to_bits(), width.to_bits()),
            None => (false, 0, 0),
        };
    FrameCacheKey {
        file_path: frame_ref.file_path.clone(),
        frame_index: frame_ref.frame_index,
        has_window_override,
        window_center_bits,
        window_width_bits,
        lut_name: session.active_lut_name.clone(),
        rotation_quarters: session.image_transform.rotation_quarters % 4,
        flip_horizontal: session.image_transform.flip_horizontal,
        flip_vertical: session.image_transform.flip_vertical,
        invert: session.image_transform.invert,
    }
}

fn transformed_image_dimensions(
    width: u32,
    height: u32,
    transform: ImageTransformState,
) -> (u32, u32) {
    if transform.rotation_quarters % 2 == 0 {
        (width, height)
    } else {
        (height, width)
    }
}

fn source_to_display_point_raw(
    point: DVec2,
    source_width: f64,
    source_height: f64,
    transform: ImageTransformState,
) -> DVec2 {
    let rotation = transform.rotation_quarters % 4;
    let (mut x, mut y, display_width, display_height) = match rotation {
        0 => (point.x, point.y, source_width, source_height),
        1 => (
            source_height - point.y,
            point.x,
            source_height,
            source_width,
        ),
        2 => (
            source_width - point.x,
            source_height - point.y,
            source_width,
            source_height,
        ),
        3 => (point.y, source_width - point.x, source_height, source_width),
        _ => unreachable!(),
    };

    if transform.flip_horizontal {
        x = display_width - x;
    }
    if transform.flip_vertical {
        y = display_height - y;
    }

    DVec2::new(x, y)
}

fn display_to_source_point_raw(
    point: DVec2,
    source_width: f64,
    source_height: f64,
    transform: ImageTransformState,
) -> DVec2 {
    let (display_width, display_height) =
        transformed_image_dimensions(source_width as u32, source_height as u32, transform);
    let mut x = point.x;
    let mut y = point.y;

    if transform.flip_horizontal {
        x = display_width as f64 - x;
    }
    if transform.flip_vertical {
        y = display_height as f64 - y;
    }

    match transform.rotation_quarters % 4 {
        0 => DVec2::new(x, y),
        1 => DVec2::new(y, source_height - x),
        2 => DVec2::new(source_width - x, source_height - y),
        3 => DVec2::new(source_width - y, x),
        _ => unreachable!(),
    }
}

fn transform_rgba(
    rgba: &[u8],
    source_width: u32,
    source_height: u32,
    transform: ImageTransformState,
) -> Vec<u8> {
    let (display_width, display_height) =
        transformed_image_dimensions(source_width, source_height, transform);
    let mut output = vec![0u8; display_width as usize * display_height as usize * 4];
    let display_width_usize = display_width as usize;
    let display_height_usize = display_height as usize;
    let source_width_usize = source_width as usize;
    let source_height_usize = source_height as usize;

    for dy in 0..display_height_usize {
        for dx in 0..display_width_usize {
            let mut rx = dx;
            let mut ry = dy;

            if transform.flip_horizontal {
                rx = display_width_usize - 1 - rx;
            }
            if transform.flip_vertical {
                ry = display_height_usize - 1 - ry;
            }

            let (sx, sy) = match transform.rotation_quarters % 4 {
                0 => (rx, ry),
                1 => (ry, source_height_usize - 1 - rx),
                2 => (source_width_usize - 1 - rx, source_height_usize - 1 - ry),
                3 => (source_width_usize - 1 - ry, rx),
                _ => unreachable!(),
            };

            let source_index = (sy * source_width_usize + sx) * 4;
            let output_index = (dy * display_width_usize + dx) * 4;
            output[output_index..output_index + 4]
                .copy_from_slice(&rgba[source_index..source_index + 4]);
            if transform.invert {
                output[output_index] = 255 - output[output_index];
                output[output_index + 1] = 255 - output[output_index + 1];
                output[output_index + 2] = 255 - output[output_index + 2];
            }
        }
    }

    output
}

fn patient_orientation_label(vector: [f64; 3]) -> String {
    let mut axes = [
        (vector[0].abs(), if vector[0] >= 0.0 { 'L' } else { 'R' }),
        (vector[1].abs(), if vector[1] >= 0.0 { 'P' } else { 'A' }),
        (vector[2].abs(), if vector[2] >= 0.0 { 'S' } else { 'I' }),
    ];
    axes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    axes.into_iter()
        .filter(|(magnitude, _)| *magnitude > 0.2)
        .take(3)
        .map(|(_, label)| label)
        .collect()
}

fn orientation_labels_for_frame(
    orientation: Option<[f64; 6]>,
    transform: ImageTransformState,
) -> (String, String, String, String) {
    let Some(iop) = orientation else {
        return (String::new(), String::new(), String::new(), String::new());
    };
    let row = [iop[0], iop[1], iop[2]];
    let col = [iop[3], iop[4], iop[5]];
    let (mut right, mut down) = match transform.rotation_quarters % 4 {
        0 => (row, col),
        1 => ([-col[0], -col[1], -col[2]], [row[0], row[1], row[2]]),
        2 => ([-row[0], -row[1], -row[2]], [-col[0], -col[1], -col[2]]),
        3 => ([col[0], col[1], col[2]], [-row[0], -row[1], -row[2]]),
        _ => unreachable!(),
    };

    if transform.flip_horizontal {
        right = [-right[0], -right[1], -right[2]];
    }
    if transform.flip_vertical {
        down = [-down[0], -down[1], -down[2]];
    }

    (
        patient_orientation_label([-down[0], -down[1], -down[2]]),
        patient_orientation_label(down),
        patient_orientation_label([-right[0], -right[1], -right[2]]),
        patient_orientation_label(right),
    )
}

fn apply_orientation_labels(
    viewer: &leaf_ui::StudyViewerWindow,
    orientation: Option<[f64; 6]>,
    transform: ImageTransformState,
) {
    let (top, bottom, left, right) = orientation_labels_for_frame(orientation, transform);
    viewer.set_orientation_top(top.into());
    viewer.set_orientation_bottom(bottom.into());
    viewer.set_orientation_left(left.into());
    viewer.set_orientation_right(right.into());
}

fn orientation_labels_for_slice_plane(plane: &SlicePlane) -> (String, String, String, String) {
    (
        patient_orientation_label([-plane.up.x, -plane.up.y, -plane.up.z]),
        patient_orientation_label([plane.up.x, plane.up.y, plane.up.z]),
        patient_orientation_label([-plane.right.x, -plane.right.y, -plane.right.z]),
        patient_orientation_label([plane.right.x, plane.right.y, plane.right.z]),
    )
}

fn apply_slice_orientation_labels(viewer: &leaf_ui::StudyViewerWindow, plane: &SlicePlane) {
    let (top, bottom, left, right) = orientation_labels_for_slice_plane(plane);
    viewer.set_orientation_top(top.into());
    viewer.set_orientation_bottom(bottom.into());
    viewer.set_orientation_left(left.into());
    viewer.set_orientation_right(right.into());
}

fn clear_orientation_labels(viewer: &leaf_ui::StudyViewerWindow) {
    viewer.set_orientation_top("".into());
    viewer.set_orientation_bottom("".into());
    viewer.set_orientation_left("".into());
    viewer.set_orientation_right("".into());
}

fn apply_quad_orientation_labels(
    viewer: &leaf_ui::StudyViewerWindow,
    kind: QuadViewportKind,
    plane: Option<&SlicePlane>,
) {
    let (top, bottom, left, right) = plane
        .map(orientation_labels_for_slice_plane)
        .unwrap_or_else(|| (String::new(), String::new(), String::new(), String::new()));
    match kind {
        QuadViewportKind::Axial => {
            viewer.set_quad_axial_orientation_top(top.into());
            viewer.set_quad_axial_orientation_bottom(bottom.into());
            viewer.set_quad_axial_orientation_left(left.into());
            viewer.set_quad_axial_orientation_right(right.into());
        }
        QuadViewportKind::Coronal => {
            viewer.set_quad_coronal_orientation_top(top.into());
            viewer.set_quad_coronal_orientation_bottom(bottom.into());
            viewer.set_quad_coronal_orientation_left(left.into());
            viewer.set_quad_coronal_orientation_right(right.into());
        }
        QuadViewportKind::Sagittal => {
            viewer.set_quad_sagittal_orientation_top(top.into());
            viewer.set_quad_sagittal_orientation_bottom(bottom.into());
            viewer.set_quad_sagittal_orientation_left(left.into());
            viewer.set_quad_sagittal_orientation_right(right.into());
        }
        QuadViewportKind::Dvr => {}
    }
}

fn clear_quad_orientation_labels(viewer: &leaf_ui::StudyViewerWindow) {
    for kind in [
        QuadViewportKind::Axial,
        QuadViewportKind::Coronal,
        QuadViewportKind::Sagittal,
    ] {
        apply_quad_orientation_labels(viewer, kind, None);
    }
}

fn apply_quad_crosshair_handle(
    viewer: &leaf_ui::StudyViewerWindow,
    kind: QuadViewportKind,
    point: Option<(f32, f32)>,
    active: bool,
) {
    let (x, y, visible) = point
        .map(|(x, y)| (x, y, true))
        .unwrap_or((0.0, 0.0, false));
    match kind {
        QuadViewportKind::Axial => {
            viewer.set_quad_axial_crosshair_x(x);
            viewer.set_quad_axial_crosshair_y(y);
            viewer.set_quad_axial_crosshair_visible(visible);
            viewer.set_quad_axial_crosshair_active(active);
        }
        QuadViewportKind::Coronal => {
            viewer.set_quad_coronal_crosshair_x(x);
            viewer.set_quad_coronal_crosshair_y(y);
            viewer.set_quad_coronal_crosshair_visible(visible);
            viewer.set_quad_coronal_crosshair_active(active);
        }
        QuadViewportKind::Sagittal => {
            viewer.set_quad_sagittal_crosshair_x(x);
            viewer.set_quad_sagittal_crosshair_y(y);
            viewer.set_quad_sagittal_crosshair_visible(visible);
            viewer.set_quad_sagittal_crosshair_active(active);
        }
        QuadViewportKind::Dvr => {}
    }
}

fn clear_quad_crosshair_handles(viewer: &leaf_ui::StudyViewerWindow) {
    for kind in [
        QuadViewportKind::Axial,
        QuadViewportKind::Coronal,
        QuadViewportKind::Sagittal,
    ] {
        apply_quad_crosshair_handle(viewer, kind, None, false);
    }
}

fn image_to_viewport_point(session: &ViewerSession, point: DVec2) -> Option<(f32, f32)> {
    let geometry = current_viewport_geometry(session)?;
    let source_width = session.active_frame_width as f64;
    let source_height = session.active_frame_height as f64;
    let display_width = session.display_frame_width as f32;
    let display_height = session.display_frame_height as f32;
    if source_width <= 0.0 || source_height <= 0.0 || display_width <= 0.0 || display_height <= 0.0
    {
        return None;
    }

    let display_point =
        source_to_display_point_raw(point, source_width, source_height, session.image_transform);
    Some((
        geometry.image_origin_x + (display_point.x as f32 / display_width) * geometry.image_width,
        geometry.image_origin_y + (display_point.y as f32 / display_height) * geometry.image_height,
    ))
}

fn viewport_to_image_point(session: &ViewerSession, x: f32, y: f32, clamp: bool) -> Option<DVec2> {
    let geometry = current_viewport_geometry(session)?;
    let source_width = session.active_frame_width as f64;
    let source_height = session.active_frame_height as f64;
    let display_width = session.display_frame_width as f32;
    let display_height = session.display_frame_height as f32;
    if source_width <= 0.0 || source_height <= 0.0 || display_width <= 0.0 || display_height <= 0.0
    {
        return None;
    }

    let mut normalized_x = (x - geometry.image_origin_x) / geometry.image_width;
    let mut normalized_y = (y - geometry.image_origin_y) / geometry.image_height;

    if clamp {
        normalized_x = normalized_x.clamp(0.0, 1.0);
        normalized_y = normalized_y.clamp(0.0, 1.0);
    } else if !(0.0..=1.0).contains(&normalized_x) || !(0.0..=1.0).contains(&normalized_y) {
        return None;
    }

    let display_point = DVec2::new(
        normalized_x as f64 * display_width as f64,
        normalized_y as f64 * display_height as f64,
    );
    let source_point = display_to_source_point_raw(
        display_point,
        source_width,
        source_height,
        session.image_transform,
    );

    Some(DVec2::new(
        source_point.x.clamp(0.0, source_width),
        source_point.y.clamp(0.0, source_height),
    ))
}

fn current_viewport_geometry(session: &ViewerSession) -> Option<ViewportGeometry> {
    displayed_image_geometry(
        session.viewport_width,
        session.viewport_height,
        session.display_frame_width,
        session.display_frame_height,
        session.viewport_scale,
        session.viewport_offset_x,
        session.viewport_offset_y,
    )
}

fn mpr_uv_from_viewport(geometry: ViewportGeometry, x: f32, y: f32) -> Option<DVec2> {
    let normalized_x = (x - geometry.image_origin_x) / geometry.image_width;
    let normalized_y = (y - geometry.image_origin_y) / geometry.image_height;
    if !(0.0..=1.0).contains(&normalized_x) || !(0.0..=1.0).contains(&normalized_y) {
        return None;
    }
    Some(DVec2::new(normalized_x as f64, normalized_y as f64))
}

fn mpr_world_point_from_viewport(session: &ViewerSession, x: f32, y: f32) -> Option<glam::DVec3> {
    if !session.volume_preview_active || session.advanced_preview_mode.is_dvr() {
        return None;
    }
    let geometry = current_viewport_geometry(session)?;
    let prepared = session
        .prepared_volumes_by_series
        .get(&session.active_series_uid)?;
    let slice_state = session
        .slice_view_state_by_series
        .get(&session.active_series_uid)
        .copied()
        .unwrap_or_default();
    let slice_uv = mpr_uv_from_viewport(geometry, x, y)?;
    let plane = slice_state.slice_plane(prepared.world_bounds());
    Some(plane.point_to_world(slice_uv))
}

fn quad_slice_view_state_for_kind(
    session: &ViewerSession,
    prepared: &PreparedVolume,
    kind: QuadViewportKind,
) -> Option<SlicePreviewState> {
    let slice_mode = kind.slice_mode()?;
    let scalar_range = prepared.scalar_range();
    let bounds = prepared.world_bounds();
    let mut state = session
        .slice_view_state_by_series
        .get(&session.active_series_uid)
        .copied()
        .unwrap_or_default();
    if state.mode != slice_mode {
        state.set_mode(slice_mode);
    }
    state.ensure_transfer_window(scalar_range.0, scalar_range.1);
    state.center_on_crosshair(bounds);
    Some(state)
}

fn quad_mpr_world_point_from_viewport(
    session: &ViewerSession,
    kind: QuadViewportKind,
    x: f32,
    y: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<glam::DVec3> {
    let preview = quad_preview(session, kind)?;
    let geometry = displayed_image_geometry(
        viewport_width,
        viewport_height,
        preview.width,
        preview.height,
        1.0,
        0.0,
        0.0,
    )?;
    let prepared = session
        .prepared_volumes_by_series
        .get(&session.active_series_uid)?;
    let slice_state = quad_slice_view_state_for_kind(session, prepared, kind)?;
    let slice_uv = mpr_uv_from_viewport(geometry, x, y)?;
    let plane = slice_state.slice_plane(prepared.world_bounds());
    Some(plane.point_to_world(slice_uv))
}

fn adjust_quad_crosshair_by_scroll(
    session: &mut ViewerSession,
    kind: QuadViewportKind,
    delta: f32,
) -> LeafResult<()> {
    let Some(slice_mode) = kind.slice_mode() else {
        return Ok(());
    };
    let prepared = session
        .prepared_volumes_by_series
        .get(&session.active_series_uid)
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    let bounds = prepared.world_bounds();
    let step = prepared.slice_scroll_step(slice_mode);
    let delta_mm = if delta < 0.0 {
        step
    } else if delta > 0.0 {
        -step
    } else {
        0.0
    };
    let slice_state = active_slice_view_state(session);
    if slice_state.mode != slice_mode {
        slice_state.set_mode(slice_mode);
    }
    let normal = slice_state.slice_plane(bounds).normal();
    let world = slice_state.crosshair_world(bounds) + normal * delta_mm;
    slice_state.center_on_world(world, bounds);
    Ok(())
}

fn displayed_image_geometry(
    viewport_width: f32,
    viewport_height: f32,
    frame_width: u32,
    frame_height: u32,
    viewport_scale: f32,
    viewport_offset_x: f32,
    viewport_offset_y: f32,
) -> Option<ViewportGeometry> {
    if viewport_width <= 0.0 || viewport_height <= 0.0 || frame_width == 0 || frame_height == 0 {
        return None;
    }

    let scaled_width = viewport_width * viewport_scale.max(0.1);
    let scaled_height = viewport_height * viewport_scale.max(0.1);
    let fit = (scaled_width / frame_width as f32).min(scaled_height / frame_height as f32);
    if !fit.is_finite() || fit <= 0.0 {
        return None;
    }

    let image_width = frame_width as f32 * fit;
    let image_height = frame_height as f32 * fit;
    Some(ViewportGeometry {
        image_origin_x: (viewport_width - image_width) / 2.0 + viewport_offset_x,
        image_origin_y: (viewport_height - image_height) / 2.0 + viewport_offset_y,
        image_width,
        image_height,
    })
}

pub(crate) fn install_viewer_tool_state(viewer: &leaf_ui::StudyViewerWindow) {
    let viewer_weak = viewer.as_weak();
    viewer.on_tool_selected(move |tool| {
        if let Some(viewer) = viewer_weak.upgrade() {
            viewer.set_active_tool(tool);
        }
    });
}

fn update_viewer_image(session: &mut ViewerSession) -> LeafResult<()> {
    if session.volume_preview_active {
        return render_or_show_volume_preview(session);
    }
    let frames = session
        .frames_by_series
        .get(&session.active_series_uid)
        .ok_or_else(|| LeafError::NoData("Series has no indexed frames".into()))?;
    let total = frames.len();
    let frame_ref = frames
        .get(session.active_frame_index)
        .cloned()
        .ok_or_else(|| LeafError::NoData("Frame index out of range".into()))?;
    let cache_key = frame_cache_key(session, &frame_ref);
    let cached = session.frame_cache.get(&cache_key).cloned();
    let cached = if let Some(cached) = cached {
        cached
    } else {
        let frame = decode_frame_with_window(
            Path::new(&frame_ref.file_path),
            frame_ref.frame_index,
            session.window_center.zip(session.window_width),
        )?;
        if !session
            .overlay_cache_by_file
            .contains_key(&frame_ref.file_path)
        {
            let overlays = load_overlays(Path::new(&frame_ref.file_path))?;
            session
                .overlay_cache_by_file
                .insert(frame_ref.file_path.clone(), overlays);
        }
        let mut rgba = to_rgba(&frame, &session.active_lut_name)?;
        if let Some(overlays) = session.overlay_cache_by_file.get(&frame_ref.file_path) {
            apply_overlays_to_rgba(&mut rgba, frame.width, frame.height, overlays);
        }
        let transformed = transform_rgba(&rgba, frame.width, frame.height, session.image_transform);
        let (display_width, display_height) =
            transformed_image_dimensions(frame.width, frame.height, session.image_transform);
        let cached = CachedFrame {
            source_width: frame.width,
            source_height: frame.height,
            display_width,
            display_height,
            rgba: transformed,
            window_center: frame.window_center,
            window_width: frame.window_width,
        };
        session.frame_cache.put(cache_key, cached.clone());
        cached
    };

    if session.default_window_center.is_none() || session.default_window_width.is_none() {
        session.default_window_center = Some(cached.window_center);
        session.default_window_width = Some(cached.window_width);
    }
    if session.window_center.is_none() || session.window_width.is_none() {
        session.window_center = Some(cached.window_center);
        session.window_width = Some(cached.window_width);
    }
    session.active_frame_width = cached.source_width;
    session.active_frame_height = cached.source_height;
    session.display_frame_width = cached.display_width;
    session.display_frame_height = cached.display_height;

    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    viewer.set_connection_status("Local imagebox".into());
    viewer.set_viewport_image(
        leaf_ui::image_from_rgba8(cached.display_width, cached.display_height, cached.rgba)
            .map_err(|error| LeafError::Render(error.to_string()))?,
    );
    viewer.set_slice_info(format!("{}/{}", session.active_frame_index + 1, total).into());
    apply_orientation_labels(
        &viewer,
        frame_ref.image_orientation_patient,
        session.image_transform,
    );
    apply_viewport_state(session)?;
    Ok(())
}

fn build_frames_by_series(
    instances_by_series: &HashMap<String, Vec<InstanceInfo>>,
) -> HashMap<String, Vec<FrameRef>> {
    instances_by_series
        .iter()
        .map(|(series_uid, instances)| {
            let mut frames = Vec::new();
            for instance in instances {
                let Some(file_path) = instance.file_path.as_ref() else {
                    continue;
                };
                let count = frame_count(Path::new(file_path)).unwrap_or(1);
                for frame_index in 0..count {
                    frames.push(FrameRef {
                        file_path: file_path.clone(),
                        frame_index: frame_index as u32,
                        image_orientation_patient: instance.image_orientation_patient,
                    });
                }
            }
            (series_uid.clone(), frames)
        })
        .collect()
}

fn apply_viewport_state(session: &ViewerSession) -> LeafResult<()> {
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;

    viewer.set_active_tool(session.active_tool);
    viewer.set_volume_preview_active(session.volume_preview_active);
    viewer.set_quad_view_active(session.volume_preview_active && session.quad_viewport_active);
    viewer.set_layout_label(layout_label(session).into());
    viewer.set_focused_quad_viewport(session.focused_quad_viewport.index());
    viewer.set_image_rotated(session.image_transform.is_rotated());
    viewer.set_image_flipped_h(session.image_transform.flip_horizontal);
    viewer.set_image_flipped_v(session.image_transform.flip_vertical);
    viewer.set_image_inverted(session.image_transform.invert);
    viewer.set_lut_label(lut_label(&session.active_lut_name).into());
    viewer.set_advanced_preview_label(session.advanced_preview_mode.label().into());
    viewer.set_quad_axial_info(quad_preview_info(session, QuadViewportKind::Axial).into());
    viewer.set_quad_coronal_info(quad_preview_info(session, QuadViewportKind::Coronal).into());
    viewer.set_quad_sagittal_info(quad_preview_info(session, QuadViewportKind::Sagittal).into());
    viewer.set_quad_dvr_info(quad_preview_info(session, QuadViewportKind::Dvr).into());
    if session.volume_preview_active && session.quad_viewport_active {
        viewer.set_quad_axial_reference_lines(quad_reference_line_model(
            session,
            QuadViewportKind::Axial,
        ));
        viewer.set_quad_coronal_reference_lines(quad_reference_line_model(
            session,
            QuadViewportKind::Coronal,
        ));
        viewer.set_quad_sagittal_reference_lines(quad_reference_line_model(
            session,
            QuadViewportKind::Sagittal,
        ));
        if let Some(prepared) = session
            .prepared_volumes_by_series
            .get(&session.active_series_uid)
        {
            let bounds = prepared.world_bounds();
            let (tile_width, tile_height) = quad_tile_max_dimensions(session);
            for kind in [
                QuadViewportKind::Axial,
                QuadViewportKind::Coronal,
                QuadViewportKind::Sagittal,
            ] {
                let plane = quad_slice_view_state_for_kind(session, prepared, kind)
                    .map(|state| state.slice_plane(bounds));
                apply_quad_orientation_labels(&viewer, kind, plane.as_ref());
                apply_quad_crosshair_handle(
                    &viewer,
                    kind,
                    quad_crosshair_viewport_point(session, kind, tile_width, tile_height),
                    quad_reference_center_highlighted(session, kind),
                );
            }
        } else {
            clear_quad_orientation_labels(&viewer);
            clear_quad_crosshair_handles(&viewer);
        }
    } else {
        viewer.set_quad_axial_reference_lines(empty_quad_reference_line_model());
        viewer.set_quad_coronal_reference_lines(empty_quad_reference_line_model());
        viewer.set_quad_sagittal_reference_lines(empty_quad_reference_line_model());
        clear_quad_orientation_labels(&viewer);
        clear_quad_crosshair_handles(&viewer);
    }
    viewer.set_volume_mode_label(
        if session.advanced_preview_mode.is_dvr() {
            volume_blend_mode_label(
                session
                    .volume_view_state_by_series
                    .get(&session.active_series_uid)
                    .map(|state| state.blend_mode)
                    .unwrap_or_default(),
            )
        } else {
            slice_projection_mode_label(
                session
                    .slice_view_state_by_series
                    .get(&session.active_series_uid)
                    .map(|state| state.projection_mode)
                    .unwrap_or_default(),
            )
        }
        .into(),
    );

    if session.volume_preview_active {
        if session.quad_viewport_active {
            if session.focused_quad_viewport.is_dvr() {
                let blend_mode = session
                    .volume_view_state_by_series
                    .get(&session.active_series_uid)
                    .map(|state| state.blend_mode)
                    .unwrap_or_default();
                if let Some(prepared) = session
                    .prepared_volumes_by_series
                    .get(&session.active_series_uid)
                {
                    let (center, width) = session
                        .volume_view_state_by_series
                        .get(&session.active_series_uid)
                        .copied()
                        .unwrap_or_default()
                        .transfer_window(prepared.scalar_range().0, prepared.scalar_range().1);
                    viewer.set_window_info(format!("Quad 3D C:{center:.0} W:{width:.0}").into());
                } else {
                    viewer.set_window_info("Quad DVR".into());
                }
                let volume_zoom = session
                    .volume_view_state_by_series
                    .get(&session.active_series_uid)
                    .map(|state| state.zoom)
                    .unwrap_or(1.0);
                viewer.set_zoom_info(format!("3D {:.0}%", volume_zoom * 100.0).into());
                viewer
                    .set_slice_info(format!("Quad {}", volume_blend_mode_label(blend_mode)).into());
            } else {
                let projection_mode = session
                    .slice_view_state_by_series
                    .get(&session.active_series_uid)
                    .map(|state| state.projection_mode)
                    .unwrap_or_default();
                if let Some(prepared) = session
                    .prepared_volumes_by_series
                    .get(&session.active_series_uid)
                {
                    let (center, width) = session
                        .slice_view_state_by_series
                        .get(&session.active_series_uid)
                        .copied()
                        .unwrap_or_default()
                        .transfer_window(prepared.scalar_range().0, prepared.scalar_range().1);
                    viewer.set_window_info(format!("Quad MPR C:{center:.0} W:{width:.0}").into());
                } else {
                    viewer.set_window_info("Quad MPR".into());
                }
                viewer.set_zoom_info("Quad linked".into());
                viewer.set_slice_info(
                    format!(
                        "Quad {} {}",
                        session.focused_quad_viewport.title(),
                        slice_projection_mode_label(projection_mode)
                    )
                    .into(),
                );
            }
            viewer.set_viewport_scale(1.0);
            viewer.set_viewport_offset_x(0.0);
            viewer.set_viewport_offset_y(0.0);
            clear_orientation_labels(&viewer);
            update_measurement_overlays(session)?;
            return Ok(());
        }

        if session.advanced_preview_mode.is_dvr() {
            let blend_mode = session
                .volume_view_state_by_series
                .get(&session.active_series_uid)
                .map(|state| state.blend_mode)
                .unwrap_or_default();
            if let Some(prepared) = session
                .prepared_volumes_by_series
                .get(&session.active_series_uid)
            {
                let (center, width) = session
                    .volume_view_state_by_series
                    .get(&session.active_series_uid)
                    .copied()
                    .unwrap_or_default()
                    .transfer_window(prepared.scalar_range().0, prepared.scalar_range().1);
                viewer.set_window_info(format!("3D C:{center:.0} W:{width:.0}").into());
            } else {
                viewer.set_window_info("DVR preview".into());
            }
            let volume_zoom = session
                .volume_view_state_by_series
                .get(&session.active_series_uid)
                .map(|state| state.zoom)
                .unwrap_or(1.0);
            viewer.set_viewport_scale(1.0);
            viewer.set_viewport_offset_x(0.0);
            viewer.set_viewport_offset_y(0.0);
            viewer.set_zoom_info(format!("3D {:.0}%", volume_zoom * 100.0).into());
            viewer.set_slice_info(format!("3D {}", volume_blend_mode_label(blend_mode)).into());
            clear_orientation_labels(&viewer);
        } else {
            let projection_mode = session
                .slice_view_state_by_series
                .get(&session.active_series_uid)
                .map(|state| state.projection_mode)
                .unwrap_or_default();
            if let Some(prepared) = session
                .prepared_volumes_by_series
                .get(&session.active_series_uid)
            {
                let bounds = prepared.world_bounds();
                let (center, width) = session
                    .slice_view_state_by_series
                    .get(&session.active_series_uid)
                    .copied()
                    .unwrap_or_default()
                    .transfer_window(prepared.scalar_range().0, prepared.scalar_range().1);
                viewer.set_window_info(format!("MPR C:{center:.0} W:{width:.0}").into());
                let slice_mode = session
                    .advanced_preview_mode
                    .slice_mode()
                    .unwrap_or_default();
                let mut state = session
                    .slice_view_state_by_series
                    .get(&session.active_series_uid)
                    .copied()
                    .unwrap_or_default();
                if state.mode != slice_mode {
                    state.set_mode(slice_mode);
                }
                state.center_on_crosshair(bounds);
                apply_slice_orientation_labels(&viewer, &state.slice_plane(bounds));
            } else {
                viewer.set_window_info("MPR preview".into());
                clear_orientation_labels(&viewer);
            }
            viewer.set_zoom_info(format!("{:.0}%", session.viewport_scale * 100.0).into());
            viewer.set_viewport_scale(session.viewport_scale);
            viewer.set_viewport_offset_x(session.viewport_offset_x);
            viewer.set_viewport_offset_y(session.viewport_offset_y);
            viewer.set_slice_info(
                format!(
                    "MPR {} {}",
                    session.advanced_preview_mode.label(),
                    slice_projection_mode_label(projection_mode)
                )
                .into(),
            );
        }
        update_measurement_overlays(session)?;
        return Ok(());
    }

    viewer.set_viewport_scale(session.viewport_scale);
    viewer.set_viewport_offset_x(session.viewport_offset_x);
    viewer.set_viewport_offset_y(session.viewport_offset_y);
    viewer.set_zoom_info(format!("{:.0}%", session.viewport_scale * 100.0).into());

    if let (Some(center), Some(width)) = (session.window_center, session.window_width) {
        viewer.set_window_info(format!("W:{width:.0} L:{center:.0}").into());
    }

    update_measurement_overlays(session)?;

    Ok(())
}

fn ensure_active_prepared_volume(session: &mut ViewerSession) -> LeafResult<String> {
    let series_uid = session.active_series_uid.clone();
    if !session.prepared_volumes_by_series.contains_key(&series_uid) {
        let file_paths = active_series_file_paths(session)?;
        let prepared = {
            let renderer = ensure_volume_renderer(session)?;
            renderer.prepare_series_volume(&file_paths, &SeriesUid(series_uid.clone()))?
        };
        session
            .prepared_volumes_by_series
            .insert(series_uid.clone(), prepared);
    }
    Ok(series_uid)
}

fn render_or_show_volume_preview(session: &mut ViewerSession) -> LeafResult<()> {
    if session.quad_viewport_active {
        render_quad_volume_preview(session)
    } else if session.advanced_preview_mode.is_dvr() {
        render_volume_preview(session, false)
    } else {
        render_slice_preview(session)
    }
}

fn render_volume_preview(session: &mut ViewerSession, interactive: bool) -> LeafResult<()> {
    let series_uid = ensure_active_prepared_volume(session)?;
    let preview_size = preview_dimensions(session);
    let scalar_range = session
        .prepared_volumes_by_series
        .get(&series_uid)
        .map(|prepared| prepared.scalar_range())
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    active_volume_view_state(session).ensure_transfer_window(scalar_range.0, scalar_range.1);
    let view_state = session
        .volume_view_state_by_series
        .get(&series_uid)
        .copied()
        .unwrap_or_default();
    if session.volume_renderer.is_none() {
        session.volume_renderer = Some(VolumePreviewRenderer::new()?);
    }
    let preview = {
        let ViewerSession {
            volume_renderer,
            prepared_volumes_by_series,
            ..
        } = session;
        let renderer = volume_renderer
            .as_mut()
            .ok_or_else(|| LeafError::Render("Volume renderer unavailable".into()))?;
        let prepared = prepared_volumes_by_series
            .get(&series_uid)
            .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
        renderer.render_prepared_preview(
            prepared,
            &view_state,
            preview_size.0,
            preview_size.1,
            interactive,
        )?
    };
    show_volume_preview(session, &preview)
}

fn render_slice_preview(session: &mut ViewerSession) -> LeafResult<()> {
    let series_uid = ensure_active_prepared_volume(session)?;
    let prepared = session
        .prepared_volumes_by_series
        .get(&series_uid)
        .cloned()
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    let scalar_range = prepared.scalar_range();
    let bounds = prepared.world_bounds();
    let slice_mode = session
        .advanced_preview_mode
        .slice_mode()
        .unwrap_or_default();
    let preview_size = slice_preview_dimensions(session, &prepared, slice_mode);
    {
        let slice_state = active_slice_view_state(session);
        if slice_state.mode != slice_mode {
            slice_state.set_mode(slice_mode);
        }
        slice_state.ensure_transfer_window(scalar_range.0, scalar_range.1);
        slice_state.center_on_crosshair(bounds);
    }
    let view_state = session
        .slice_view_state_by_series
        .get(&series_uid)
        .copied()
        .unwrap_or_else(|| {
            let mut state = SlicePreviewState::default();
            state.set_mode(slice_mode);
            state
        });
    if session.volume_renderer.is_none() {
        session.volume_renderer = Some(VolumePreviewRenderer::new()?);
    }
    let preview = {
        let renderer = session
            .volume_renderer
            .as_mut()
            .ok_or_else(|| LeafError::Render("Volume renderer unavailable".into()))?;
        renderer.render_prepared_slice_preview(
            &prepared,
            &view_state,
            preview_size.0,
            preview_size.1,
            true,
        )?
    };
    show_volume_preview(session, &preview)
}

fn render_quad_preview_for_kind(
    session: &mut ViewerSession,
    kind: QuadViewportKind,
    interactive: bool,
) -> LeafResult<AdvancedViewportPreview> {
    let series_uid = ensure_active_prepared_volume(session)?;
    let prepared = session
        .prepared_volumes_by_series
        .get(&series_uid)
        .cloned()
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    let (tile_width, tile_height) = quad_tile_max_dimensions(session);

    if kind.is_dvr() {
        let preview_size = preview_dimensions_for_viewport(tile_width, tile_height);
        let scalar_range = prepared.scalar_range();
        active_volume_view_state(session).ensure_transfer_window(scalar_range.0, scalar_range.1);
        let view_state = session
            .volume_view_state_by_series
            .get(&series_uid)
            .copied()
            .unwrap_or_default();
        let preview = {
            let renderer = ensure_volume_renderer(session)?;
            renderer.render_prepared_preview(
                &prepared,
                &view_state,
                preview_size.0,
                preview_size.1,
                interactive,
            )?
        };
        return Ok(AdvancedViewportPreview {
            width: preview.width,
            height: preview.height,
            image: leaf_ui::image_from_rgba8(preview.width, preview.height, preview.rgba)
                .map_err(|error| LeafError::Render(error.to_string()))?,
            info: format!("DVR {}", volume_blend_mode_label(view_state.blend_mode)),
        });
    }

    let slice_state = quad_slice_view_state_for_kind(session, &prepared, kind)
        .ok_or_else(|| LeafError::Render("Quad MPR state unavailable".into()))?;
    let preview_size = slice_preview_dimensions_for_viewport(
        tile_width,
        tile_height,
        &prepared,
        kind.slice_mode().unwrap_or_default(),
    );
    let preview = {
        let renderer = ensure_volume_renderer(session)?;
        renderer.render_prepared_slice_preview(
            &prepared,
            &slice_state,
            preview_size.0,
            preview_size.1,
            false,
        )?
    };
    Ok(AdvancedViewportPreview {
        width: preview.width,
        height: preview.height,
        image: leaf_ui::image_from_rgba8(preview.width, preview.height, preview.rgba)
            .map_err(|error| LeafError::Render(error.to_string()))?,
        info: quad_mpr_preview_info(kind, slice_state),
    })
}

fn apply_quad_preview_to_viewer(
    viewer: &leaf_ui::StudyViewerWindow,
    kind: QuadViewportKind,
    preview: &AdvancedViewportPreview,
) {
    match kind {
        QuadViewportKind::Axial => {
            viewer.set_quad_axial_image(preview.image.clone());
            viewer.set_quad_axial_info(preview.info.clone().into());
        }
        QuadViewportKind::Coronal => {
            viewer.set_quad_coronal_image(preview.image.clone());
            viewer.set_quad_coronal_info(preview.info.clone().into());
        }
        QuadViewportKind::Sagittal => {
            viewer.set_quad_sagittal_image(preview.image.clone());
            viewer.set_quad_sagittal_info(preview.info.clone().into());
        }
        QuadViewportKind::Dvr => {
            viewer.set_quad_dvr_image(preview.image.clone());
            viewer.set_quad_dvr_info(preview.info.clone().into());
        }
    }
}

fn render_quad_single_preview(
    session: &mut ViewerSession,
    kind: QuadViewportKind,
    interactive: bool,
) -> LeafResult<()> {
    let preview = render_quad_preview_for_kind(session, kind, interactive)?;
    session.quad_previews_by_kind.insert(kind, preview.clone());
    session.volume_preview_active = true;
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    apply_quad_preview_to_viewer(&viewer, kind, &preview);
    viewer.set_connection_status(
        "Local imagebox | Quad MPR/DVR (select tile, wheel=slice/zoom, right-click=crosshair)"
            .into(),
    );
    rebuild_quad_reference_lines(session)?;
    apply_viewport_state(session)?;
    update_measurements_model(session)?;
    Ok(())
}

fn render_quad_mpr_previews(session: &mut ViewerSession) -> LeafResult<()> {
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    for kind in [
        QuadViewportKind::Axial,
        QuadViewportKind::Coronal,
        QuadViewportKind::Sagittal,
    ] {
        let preview = render_quad_preview_for_kind(session, kind, false)?;
        apply_quad_preview_to_viewer(&viewer, kind, &preview);
        session.quad_previews_by_kind.insert(kind, preview);
    }
    session.volume_preview_active = true;
    viewer.set_connection_status(
        "Local imagebox | Quad MPR/DVR (select tile, wheel=slice/zoom, right-click=crosshair)"
            .into(),
    );
    rebuild_quad_reference_lines(session)?;
    apply_viewport_state(session)?;
    update_measurements_model(session)?;
    Ok(())
}

fn render_quad_volume_preview(session: &mut ViewerSession) -> LeafResult<()> {
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    for kind in QuadViewportKind::ALL {
        let preview = render_quad_preview_for_kind(session, kind, false)?;
        apply_quad_preview_to_viewer(&viewer, kind, &preview);
        session.quad_previews_by_kind.insert(kind, preview);
    }
    session.volume_preview_active = true;
    viewer.set_connection_status(
        "Local imagebox | Quad MPR/DVR (select tile, wheel=slice/zoom, right-click=crosshair)"
            .into(),
    );
    rebuild_quad_reference_lines(session)?;
    apply_viewport_state(session)?;
    update_measurements_model(session)?;
    Ok(())
}

fn ensure_volume_renderer(session: &mut ViewerSession) -> LeafResult<&mut VolumePreviewRenderer> {
    if session.volume_renderer.is_none() {
        session.volume_renderer = Some(VolumePreviewRenderer::new()?);
    }
    session
        .volume_renderer
        .as_mut()
        .ok_or_else(|| LeafError::Render("Volume renderer unavailable".into()))
}

fn active_series_file_paths(session: &ViewerSession) -> LeafResult<Vec<String>> {
    let instances = session
        .instances_by_series
        .get(&session.active_series_uid)
        .ok_or_else(|| LeafError::NoData("Series has no instances".into()))?;
    let mut unique_paths = BTreeSet::new();
    for instance in instances {
        if let Some(file_path) = instance.file_path.as_ref() {
            unique_paths.insert(file_path.clone());
        }
    }
    let file_paths = unique_paths.into_iter().collect::<Vec<_>>();
    if file_paths.is_empty() {
        return Err(LeafError::NoData(
            "Series has no local files for volume assembly".into(),
        ));
    }
    Ok(file_paths)
}

fn preview_dimensions_for_viewport(viewport_width: f32, viewport_height: f32) -> (u32, u32) {
    let mut width = if viewport_width > 0.0 {
        viewport_width.round() as u32
    } else {
        768
    };
    let mut height = if viewport_height > 0.0 {
        viewport_height.round() as u32
    } else {
        768
    };

    width = width.max(256);
    height = height.max(256);

    let target_max_side = 640;
    let max_side = width.max(height);
    if max_side > target_max_side {
        let scale = target_max_side as f32 / max_side as f32;
        width = (width as f32 * scale).round().max(256.0) as u32;
        height = (height as f32 * scale).round().max(256.0) as u32;
    }

    (width, height)
}

fn preview_dimensions(session: &ViewerSession) -> (u32, u32) {
    preview_dimensions_for_viewport(session.viewport_width, session.viewport_height)
}

fn fit_dimensions_to_aspect(max_width: u32, max_height: u32, aspect_ratio: f64) -> (u32, u32) {
    let safe_aspect = aspect_ratio.max(0.1);
    let mut width = max_width.max(128) as f64;
    let mut height = max_height.max(128) as f64;
    if width / height > safe_aspect {
        width = height * safe_aspect;
    } else {
        height = width / safe_aspect;
    }
    (
        width.round().max(128.0) as u32,
        height.round().max(128.0) as u32,
    )
}

fn slice_preview_dimensions(
    session: &ViewerSession,
    prepared: &PreparedVolume,
    mode: SlicePreviewMode,
) -> (u32, u32) {
    slice_preview_dimensions_for_viewport(
        session.viewport_width,
        session.viewport_height,
        prepared,
        mode,
    )
}

fn slice_preview_dimensions_for_viewport(
    viewport_width: f32,
    viewport_height: f32,
    prepared: &PreparedVolume,
    mode: SlicePreviewMode,
) -> (u32, u32) {
    let (max_width, max_height) = preview_dimensions_for_viewport(viewport_width, viewport_height);
    let size = prepared.world_bounds().size();
    let (plane_width, plane_height) = match mode {
        SlicePreviewMode::Axial => (size.x.max(1.0), size.y.max(1.0)),
        SlicePreviewMode::Coronal => (size.x.max(1.0), size.z.max(1.0)),
        SlicePreviewMode::Sagittal => (size.y.max(1.0), size.z.max(1.0)),
    };
    fit_dimensions_to_aspect(max_width, max_height, plane_width / plane_height)
}

fn show_volume_preview(
    session: &mut ViewerSession,
    preview: &VolumePreviewImage,
) -> LeafResult<()> {
    session.volume_preview_active = true;
    session.active_frame_width = preview.width;
    session.active_frame_height = preview.height;
    session.display_frame_width = preview.width;
    session.display_frame_height = preview.height;
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    viewer.set_viewport_image(
        leaf_ui::image_from_rgba8(preview.width, preview.height, preview.rgba.clone())
            .map_err(|error| LeafError::Render(error.to_string()))?,
    );
    if session.advanced_preview_mode.is_dvr() {
        viewer.set_connection_status(
            "Local imagebox | DVR preview (drag=orbit, Pan/Zoom tools=3D)".into(),
        );
        viewer.set_window_info("DVR preview".into());
        viewer.set_slice_info("3D".into());
    } else {
        viewer.set_connection_status(
            "Local imagebox | MPR preview (wheel=slice, W/L/Pan/Zoom tools active)".into(),
        );
        viewer.set_window_info("MPR preview".into());
        viewer.set_slice_info(format!("MPR {}", session.advanced_preview_mode.label()).into());
    }
    apply_viewport_state(session)?;
    update_measurements_model(session)?;
    Ok(())
}

fn reset_viewport_state(session: &mut ViewerSession, clear_defaults: bool) {
    session.viewport_scale = 1.0;
    session.viewport_offset_x = 0.0;
    session.viewport_offset_y = 0.0;
    session.image_transform = ImageTransformState::default();
    session.active_lut_name = DEFAULT_LUT_NAME.to_string();
    session.drag_state = None;
    session.volume_drag_state = None;

    if clear_defaults {
        session.window_center = None;
        session.window_width = None;
        session.default_window_center = None;
        session.default_window_width = None;
    } else {
        session.window_center = session.default_window_center;
        session.window_width = session.default_window_width;
    }
}

fn sort_instances_for_stack(instances: &mut [InstanceInfo]) {
    if instances.len() <= 1 {
        return;
    }

    hydrate_instance_geometry(instances);

    let mut reference_candidates = instances.iter().collect::<Vec<_>>();
    reference_candidates.sort_by(|a, b| compare_instances_by_fallback(a, b));

    let Some(reference) = reference_candidates.get(reference_candidates.len() / 2) else {
        return;
    };

    let Some(reference_iop) = reference.image_orientation_patient else {
        instances.sort_by(compare_instances_by_fallback);
        return;
    };
    let Some(reference_ipp) = reference.image_position_patient else {
        instances.sort_by(compare_instances_by_fallback);
        return;
    };

    if !instances.iter().all(|instance| {
        instance
            .image_position_patient
            .zip(instance.image_orientation_patient)
            .map(|(_, iop)| same_orientation(&iop, &reference_iop))
            .unwrap_or(false)
    }) {
        instances.sort_by(compare_instances_by_fallback);
        return;
    }

    let scan_axis_normal = cross_product(
        [reference_iop[0], reference_iop[1], reference_iop[2]],
        [reference_iop[3], reference_iop[4], reference_iop[5]],
    );

    if vector_length(scan_axis_normal) <= 1e-6 {
        instances.sort_by(compare_instances_by_fallback);
        return;
    }

    instances.sort_by(|a, b| {
        let a_distance = a
            .image_position_patient
            .map(|ipp| slice_distance(reference_ipp, ipp, scan_axis_normal));
        let b_distance = b
            .image_position_patient
            .map(|ipp| slice_distance(reference_ipp, ipp, scan_axis_normal));

        match (a_distance, b_distance) {
            (Some(a_distance), Some(b_distance)) => b_distance
                .partial_cmp(&a_distance)
                .unwrap_or(Ordering::Equal)
                .then_with(|| compare_instances_by_fallback(a, b)),
            _ => compare_instances_by_fallback(a, b),
        }
    });
}

fn hydrate_instance_geometry(instances: &mut [InstanceInfo]) {
    for instance in instances.iter_mut() {
        if instance.image_position_patient.is_some() && instance.image_orientation_patient.is_some()
        {
            continue;
        }

        let Some(file_path) = instance.file_path.as_ref() else {
            continue;
        };

        let Ok((image_position_patient, image_orientation_patient)) =
            read_instance_geometry(Path::new(file_path))
        else {
            continue;
        };

        if instance.image_position_patient.is_none() {
            instance.image_position_patient = image_position_patient;
        }
        if instance.image_orientation_patient.is_none() {
            instance.image_orientation_patient = image_orientation_patient;
        }
    }
}

fn compare_instances_by_fallback(a: &InstanceInfo, b: &InstanceInfo) -> Ordering {
    let a_number = a.instance_number.unwrap_or(i32::MAX);
    let b_number = b.instance_number.unwrap_or(i32::MAX);

    a_number
        .cmp(&b_number)
        .then_with(|| a.sop_instance_uid.0.cmp(&b.sop_instance_uid.0))
}

fn same_orientation(a: &[f64; 6], b: &[f64; 6]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(lhs, rhs)| (lhs - rhs).abs() <= 1e-4)
}

fn cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vector_length(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn slice_distance(reference_ipp: [f64; 3], image_ipp: [f64; 3], scan_axis_normal: [f64; 3]) -> f64 {
    let delta = [
        reference_ipp[0] - image_ipp[0],
        reference_ipp[1] - image_ipp[1],
        reference_ipp[2] - image_ipp[2],
    ];

    delta[0] * scan_axis_normal[0] + delta[1] * scan_axis_normal[1] + delta[2] * scan_axis_normal[2]
}

fn to_rgba(frame: &leaf_dicom::pixel::DecodedFrame, lut_name: &str) -> LeafResult<Vec<u8>> {
    let expected = frame.width as usize * frame.height as usize;
    match frame.channels {
        1 => {
            if frame.pixels.len() != expected {
                return Err(LeafError::Render("Unexpected grayscale frame size".into()));
            }
            let color_lut = resolve_color_lut(lut_name);
            let mut rgba = Vec::with_capacity(expected * 4);
            for value in &frame.pixels {
                rgba.extend_from_slice(&color_lut.table[*value as usize]);
            }
            Ok(rgba)
        }
        3 => {
            if frame.pixels.len() != expected * 3 {
                return Err(LeafError::Render("Unexpected RGB frame size".into()));
            }
            let mut rgba = Vec::with_capacity(expected * 4);
            for chunk in frame.pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            Ok(rgba)
        }
        4 => {
            if frame.pixels.len() != expected * 4 {
                return Err(LeafError::Render("Unexpected RGBA frame size".into()));
            }
            Ok(frame.pixels.clone())
        }
        channels => Err(LeafError::Render(format!(
            "Unsupported frame channel count: {channels}"
        ))),
    }
}

/// Generate a THUMB_SIZE×THUMB_SIZE RGBA thumbnail from the middle instance of a series.
fn generate_thumbnail_rgba(instances: &[InstanceInfo]) -> Option<Vec<u8>> {
    if instances.is_empty() {
        return None;
    }
    let mid = instances.len() / 2;
    let file_path = instances[mid].file_path.as_ref()?;
    let frame = decode_frame_with_window(Path::new(file_path), 0, None).ok()?;
    if frame.width == 0 || frame.height == 0 {
        return None;
    }

    let src_w = frame.width as usize;
    let src_h = frame.height as usize;
    let channels = frame.channels as usize;
    let mut rgba = vec![0u8; THUMB_SIZE * THUMB_SIZE * 4];

    for ty in 0..THUMB_SIZE {
        for tx in 0..THUMB_SIZE {
            let sx = (tx * src_w / THUMB_SIZE).min(src_w - 1);
            let sy = (ty * src_h / THUMB_SIZE).min(src_h - 1);
            let src_idx = (sy * src_w + sx) * channels;
            let dst_idx = (ty * THUMB_SIZE + tx) * 4;

            if channels == 1 {
                let v = frame.pixels[src_idx];
                rgba[dst_idx] = v;
                rgba[dst_idx + 1] = v;
                rgba[dst_idx + 2] = v;
                rgba[dst_idx + 3] = 255;
            } else if channels >= 3 {
                rgba[dst_idx] = frame.pixels[src_idx];
                rgba[dst_idx + 1] = frame.pixels[src_idx + 1];
                rgba[dst_idx + 2] = frame.pixels[src_idx + 2];
                rgba[dst_idx + 3] = 255;
            }
        }
    }

    Some(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use leaf_core::domain::{SeriesUid, SopInstanceUid, StudyUid};
    use leaf_tools::measurement::Measurement;

    fn test_imagebox() -> Rc<Imagebox> {
        let path = std::env::temp_dir().join("pacsleaf_test_viewer.redb");
        Rc::new(Imagebox::open(&path).expect("Failed to open test imagebox"))
    }

    fn instance(
        sop_uid: &str,
        instance_number: Option<i32>,
        ipp: Option<[f64; 3]>,
        iop: Option<[f64; 6]>,
    ) -> InstanceInfo {
        InstanceInfo {
            sop_instance_uid: SopInstanceUid(sop_uid.to_string()),
            series_uid: SeriesUid("series".to_string()),
            study_uid: StudyUid("study".to_string()),
            sop_class_uid: String::new(),
            instance_number,
            image_position_patient: ipp,
            image_orientation_patient: iop,
            transfer_syntax_uid: String::new(),
            file_path: None,
        }
    }

    #[test]
    fn sorts_instances_by_patient_position_when_geometry_is_available() {
        let iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut instances = vec![
            instance("3", Some(30), Some([0.0, 0.0, 2.0]), Some(iop)),
            instance("1", Some(10), Some([0.0, 0.0, 0.0]), Some(iop)),
            instance("2", Some(20), Some([0.0, 0.0, 1.0]), Some(iop)),
        ];

        sort_instances_for_stack(&mut instances);

        let ordered_numbers = instances
            .iter()
            .map(|instance| instance.instance_number.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ordered_numbers, vec![10, 20, 30]);
    }

    #[test]
    fn falls_back_to_instance_number_when_geometry_is_missing() {
        let mut instances = vec![
            instance("3", Some(30), None, None),
            instance("1", Some(10), None, None),
            instance("2", Some(20), None, None),
        ];

        sort_instances_for_stack(&mut instances);

        let ordered_numbers = instances
            .iter()
            .map(|instance| instance.instance_number.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ordered_numbers, vec![10, 20, 30]);
    }

    #[test]
    fn maps_viewport_points_into_image_space_with_contain_fit() {
        let geometry = displayed_image_geometry(800.0, 600.0, 512, 256, 1.0, 0.0, 0.0).unwrap();
        assert!((geometry.image_origin_y - 100.0).abs() < 0.001);
        assert!((geometry.image_width - 800.0).abs() < 0.001);
        assert!((geometry.image_height - 400.0).abs() < 0.001);

        let normalized_x = (400.0 - geometry.image_origin_x) / geometry.image_width;
        let normalized_y = (300.0 - geometry.image_origin_y) / geometry.image_height;
        let image_point = DVec2::new(normalized_x as f64 * 512.0, normalized_y as f64 * 256.0);

        assert!((image_point.x - 256.0).abs() < 0.001);
        assert!((image_point.y - 128.0).abs() < 0.001);
    }

    #[test]
    fn mpr_viewport_mapping_uses_top_down_slice_uv_space() {
        let geometry = displayed_image_geometry(800.0, 600.0, 512, 256, 1.0, 0.0, 0.0).unwrap();

        let top_left = mpr_uv_from_viewport(geometry, 0.0, 100.0).unwrap();
        assert!((top_left.x - 0.0).abs() < 0.001);
        assert!((top_left.y - 0.0).abs() < 0.001);

        let bottom_right = mpr_uv_from_viewport(geometry, 800.0, 500.0).unwrap();
        assert!((bottom_right.x - 1.0).abs() < 0.001);
        assert!((bottom_right.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn transform_mapping_round_trips_for_rotated_flipped_images() {
        let transform = ImageTransformState {
            rotation_quarters: 1,
            flip_horizontal: true,
            flip_vertical: false,
            invert: false,
        };
        let source = DVec2::new(128.0, 64.0);
        let display = source_to_display_point_raw(source, 512.0, 256.0, transform);
        let reconstructed = display_to_source_point_raw(display, 512.0, 256.0, transform);

        assert!((reconstructed.x - source.x).abs() < 0.001);
        assert!((reconstructed.y - source.y).abs() < 0.001);
    }

    #[test]
    fn transformed_dimensions_swap_for_quarter_turns() {
        let transform = ImageTransformState {
            rotation_quarters: 1,
            ..ImageTransformState::default()
        };
        assert_eq!(
            transformed_image_dimensions(512, 256, transform),
            (256, 512)
        );
        assert_eq!(
            transformed_image_dimensions(512, 256, ImageTransformState::default()),
            (512, 256)
        );
    }

    #[test]
    fn preview_dimensions_keep_output_size_stable() {
        let session = ViewerSession {
            viewer: slint::Weak::default(),
            imagebox: test_imagebox(),
            series: Vec::new(),
            instances_by_series: HashMap::new(),
            frames_by_series: HashMap::new(),
            measurements_by_series: HashMap::new(),
            thumbnails_by_series: HashMap::new(),
            overlay_cache_by_file: HashMap::new(),
            active_series_uid: String::new(),
            active_frame_index: 0,
            measurement_panel_visible: false,
            volume_preview_active: false,
            quad_viewport_active: false,
            advanced_preview_mode: AdvancedPreviewMode::default(),
            focused_quad_viewport: AdvancedPreviewMode::default().quad_viewport(),
            active_tool: leaf_ui::ViewerTool::WindowLevel,
            viewport_scale: 1.0,
            viewport_offset_x: 0.0,
            viewport_offset_y: 0.0,
            window_center: None,
            window_width: None,
            default_window_center: None,
            default_window_width: None,
            viewport_width: 1400.0,
            viewport_height: 900.0,
            active_frame_width: 0,
            active_frame_height: 0,
            display_frame_width: 0,
            display_frame_height: 0,
            image_transform: ImageTransformState::default(),
            active_lut_name: DEFAULT_LUT_NAME.to_string(),
            frame_cache: LruCache::new(
                NonZeroUsize::new(FRAME_CACHE_CAPACITY).expect("frame cache capacity must be > 0"),
            ),
            selected_measurement_id: None,
            draft_measurement: None,
            handle_drag: None,
            drag_state: None,
            volume_drag_state: None,
            volume_renderer: None,
            prepared_volumes_by_series: HashMap::new(),
            volume_view_state_by_series: HashMap::new(),
            slice_view_state_by_series: HashMap::new(),
            quad_previews_by_kind: HashMap::new(),
            quad_reference_lines_by_kind: HashMap::new(),
            quad_reference_hover: None,
            quad_reference_drag: None,
        };

        let preview = preview_dimensions(&session);

        assert_eq!(preview.0.max(preview.1), 640);
    }

    #[test]
    fn fit_dimensions_preserves_requested_aspect_ratio() {
        let fitted = fit_dimensions_to_aspect(640, 480, 2.0);
        assert_eq!(fitted, (640, 320));

        let fitted_tall = fit_dimensions_to_aspect(640, 480, 0.5);
        assert_eq!(fitted_tall, (240, 480));
    }

    #[test]
    fn formats_line_measurement_values_in_millimeters() {
        let measurement =
            Measurement::line("series", 0, DVec2::new(0.0, 0.0), DVec2::new(3.0, 4.0));

        assert_eq!(measurement_value_text(&measurement, (1.0, 1.0)), "5.00 mm");
    }

    #[test]
    fn grayscale_lut_maps_pixels_to_false_color() {
        let frame = leaf_dicom::pixel::DecodedFrame {
            width: 1,
            height: 1,
            pixels: vec![255],
            channels: 1,
            window_center: 0.0,
            window_width: 0.0,
        };

        assert_eq!(
            to_rgba(&frame, "hot_iron").expect("hot iron LUT should apply"),
            vec![255, 255, 255, 255]
        );
        assert_eq!(
            to_rgba(&frame, "grayscale_inverted").expect("inverted grayscale LUT should apply"),
            vec![0, 0, 0, 255]
        );
    }

    #[test]
    fn overlays_are_composited_in_image_space() {
        let mut rgba = vec![0u8; 4 * 4 * 4];
        let overlay = OverlayBitmap {
            rows: 2,
            columns: 2,
            origin: (2, 2),
            bitmap: vec![1, 0, 0, 1],
        };

        apply_overlays_to_rgba(&mut rgba, 4, 4, &[overlay]);

        let row_stride = 4 * 4;
        let top_left = row_stride + 4;
        let bottom_right = row_stride * 2 + 8;
        let top_right = row_stride + 8;

        assert_eq!(&rgba[top_left..top_left + 4], &OVERLAY_COLOR);
        assert_eq!(&rgba[bottom_right..bottom_right + 4], &OVERLAY_COLOR);
        assert_eq!(&rgba[top_right..top_right + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn volume_blend_mode_cycles_through_supported_modes() {
        assert_eq!(
            next_volume_blend_mode(VolumeBlendMode::Composite),
            VolumeBlendMode::MaximumIntensity
        );
        assert_eq!(
            next_volume_blend_mode(VolumeBlendMode::MaximumIntensity),
            VolumeBlendMode::MinimumIntensity
        );
        assert_eq!(
            next_volume_blend_mode(VolumeBlendMode::MinimumIntensity),
            VolumeBlendMode::AverageIntensity
        );
        assert_eq!(
            next_volume_blend_mode(VolumeBlendMode::AverageIntensity),
            VolumeBlendMode::Composite
        );
    }

    #[test]
    fn quad_reference_lines_pass_through_shared_crosshair() {
        let plane = SlicePlane::new(DVec3::ZERO, DVec3::X, DVec3::Y, 10.0, 10.0);
        let other_plane =
            SlicePlane::new(DVec3::new(2.0, 0.0, 0.0), DVec3::Y, DVec3::Z, 10.0, 10.0);
        let shared_world = DVec3::new(2.0, 3.0, 0.0);
        let geometry = ViewportGeometry {
            image_origin_x: 0.0,
            image_origin_y: 0.0,
            image_width: 100.0,
            image_height: 100.0,
        };

        let overlay = quad_reference_line_for_plane(
            &plane,
            &other_plane,
            QuadViewportKind::Coronal,
            shared_world,
            geometry,
        )
        .expect("reference line should exist");
        let (shared_x, shared_y) = quad_world_to_viewport_point(&plane, geometry, shared_world);

        assert!(
            point_to_segment_distance_sq(
                shared_x,
                shared_y,
                overlay.start_x,
                overlay.start_y,
                overlay.end_x,
                overlay.end_y,
            ) < 1.0e-3
        );
    }

    #[test]
    fn slice_plane_orientation_labels_follow_plane_axes() {
        let plane = SlicePlane::new(DVec3::ZERO, DVec3::X, DVec3::Y, 10.0, 10.0);
        let (top, bottom, left, right) = orientation_labels_for_slice_plane(&plane);

        assert_eq!(top, "A");
        assert_eq!(bottom, "P");
        assert_eq!(left, "R");
        assert_eq!(right, "L");
    }

    #[test]
    fn slice_plane_angles_follow_plane_basis_instead_of_screen_y() {
        let plane = SlicePlane::new(DVec3::ZERO, DVec3::X, -DVec3::Z, 10.0, 10.0);
        let right = DVec3::X;
        let top = DVec3::Z;
        let bottom = -DVec3::Z;

        let angle_for = |world: DVec3| {
            let offset = world - DVec3::ZERO;
            offset.dot(plane.up).atan2(offset.dot(plane.right))
        };

        assert!((angle_for(right) - 0.0).abs() < 1.0e-6);
        assert!((angle_for(top) + std::f64::consts::FRAC_PI_2).abs() < 1.0e-6);
        assert!((angle_for(bottom) - std::f64::consts::FRAC_PI_2).abs() < 1.0e-6);
    }

    #[test]
    fn normalized_angle_delta_chooses_shortest_rotation() {
        let delta = normalized_angle_delta(-std::f64::consts::PI + 0.1, std::f64::consts::PI - 0.1);
        assert!((delta - 0.2).abs() < 1.0e-6);
    }
}
