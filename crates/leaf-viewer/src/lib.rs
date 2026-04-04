//! Shared viewer-core utilities for pacsleaf hosts.
//!
//! This crate intentionally avoids any UI-framework dependency so its
//! rendering/math helpers and session state can be reused by the native Slint
//! host, future streamers, and other integrations.

use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
    num::NonZeroUsize,
    path::Path,
    rc::Rc,
};

use glam::{DQuat, DVec2, DVec3};
use leaf_core::{
    domain::{InstanceInfo, SeriesInfo, SeriesUid},
    error::{LeafError, LeafResult},
};
use leaf_db::imagebox::Imagebox;
use leaf_dicom::{
    metadata::read_instance_geometry,
    overlay::OverlayBitmap,
    pixel::{decode_frame_for_measurements, frame_count, DecodedFrame, MeasurementFrame},
};
use leaf_render::{
    lut::ColorLut, PreparedVolume, SlicePlane, SlicePreviewMode, SlicePreviewState,
    SliceProjectionMode, VolumeBlendMode, VolumePreviewImage, VolumePreviewRenderer,
    VolumeViewState,
};
use leaf_tools::measurement::{Measurement, MeasurementImage, MeasurementKind, MeasurementValue};
use lru::LruCache;
use tracing::info;

pub const THUMB_SIZE: usize = 64;
pub const FRAME_CACHE_CAPACITY: usize = 32;
pub const DEFAULT_LUT_NAME: &str = "grayscale";
pub const LUT_PRESETS: [(&str, &str); 4] = [
    ("Gray", "grayscale"),
    ("Hot", "hot_iron"),
    ("Bone", "bone"),
    ("InvG", "grayscale_inverted"),
];
pub const OVERLAY_COLOR: [u8; 4] = [255, 196, 0, 255];

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct ImageTransformState {
    pub rotation_quarters: u8,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub invert: bool,
}

impl ImageTransformState {
    pub fn is_rotated(self) -> bool {
        self.rotation_quarters % 4 != 0
    }
}

/// Mirrors the Slint `ViewerTool` enum without a Slint dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ViewerTool {
    #[default]
    WindowLevel,
    Pan,
    Zoom,
    Scroll,
    Line,
    Angle,
    RectangleRoi,
    EllipseRoi,
    Annotation,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum AdvancedPreviewMode {
    #[default]
    Dvr,
    Axial,
    Coronal,
    Sagittal,
}

impl AdvancedPreviewMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dvr => "DVR",
            Self::Axial => "Ax",
            Self::Coronal => "Co",
            Self::Sagittal => "Sa",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Dvr => Self::Axial,
            Self::Axial => Self::Coronal,
            Self::Coronal => Self::Sagittal,
            Self::Sagittal => Self::Dvr,
        }
    }

    pub fn slice_mode(self) -> Option<SlicePreviewMode> {
        match self {
            Self::Dvr => None,
            Self::Axial => Some(SlicePreviewMode::Axial),
            Self::Coronal => Some(SlicePreviewMode::Coronal),
            Self::Sagittal => Some(SlicePreviewMode::Sagittal),
        }
    }

    pub fn is_dvr(self) -> bool {
        matches!(self, Self::Dvr)
    }

    pub fn quad_viewport(self) -> QuadViewportKind {
        match self {
            Self::Axial => QuadViewportKind::Axial,
            Self::Coronal => QuadViewportKind::Coronal,
            Self::Sagittal => QuadViewportKind::Sagittal,
            Self::Dvr => QuadViewportKind::Dvr,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuadViewportKind {
    Axial,
    Coronal,
    Sagittal,
    Dvr,
}

impl QuadViewportKind {
    pub const ALL: [Self; 4] = [Self::Axial, Self::Coronal, Self::Sagittal, Self::Dvr];

    pub fn from_index(index: i32) -> Option<Self> {
        match index {
            0 => Some(Self::Axial),
            1 => Some(Self::Coronal),
            2 => Some(Self::Sagittal),
            3 => Some(Self::Dvr),
            _ => None,
        }
    }

    pub fn index(self) -> i32 {
        match self {
            Self::Axial => 0,
            Self::Coronal => 1,
            Self::Sagittal => 2,
            Self::Dvr => 3,
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Axial => "Axial",
            Self::Coronal => "Coronal",
            Self::Sagittal => "Sagittal",
            Self::Dvr => "DVR",
        }
    }

    pub fn advanced_preview_mode(self) -> AdvancedPreviewMode {
        match self {
            Self::Axial => AdvancedPreviewMode::Axial,
            Self::Coronal => AdvancedPreviewMode::Coronal,
            Self::Sagittal => AdvancedPreviewMode::Sagittal,
            Self::Dvr => AdvancedPreviewMode::Dvr,
        }
    }

    pub fn slice_mode(self) -> Option<SlicePreviewMode> {
        self.advanced_preview_mode().slice_mode()
    }

    pub fn is_dvr(self) -> bool {
        matches!(self, Self::Dvr)
    }

    pub fn linked_mpr_views(self) -> [Self; 2] {
        match self {
            Self::Axial => [Self::Coronal, Self::Sagittal],
            Self::Coronal => [Self::Axial, Self::Sagittal],
            Self::Sagittal => [Self::Axial, Self::Coronal],
            Self::Dvr => [Self::Axial, Self::Coronal],
        }
    }
}

#[derive(Clone)]
pub struct FrameRef {
    pub file_path: String,
    pub frame_index: u32,
    pub image_orientation_patient: Option<[f64; 6]>,
}

pub struct LoadedSeriesData {
    pub series_uid: String,
    pub instances: Vec<InstanceInfo>,
    pub frames: Vec<FrameRef>,
    pub measurements: Vec<Measurement>,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewportDragState {
    pub origin_x: f32,
    pub origin_y: f32,
    pub start_offset_x: f32,
    pub start_offset_y: f32,
    pub start_scale: f32,
    pub start_window_center: f64,
    pub start_window_width: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct VolumeDragState {
    pub origin_x: f32,
    pub origin_y: f32,
    pub button: i32,
    pub start_view_state: VolumeViewState,
}

#[derive(Clone)]
pub struct RgbaPreview {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub info: String,
}

pub type AdvancedViewportPreview = RgbaPreview;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuadReferenceTarget {
    Center,
    TranslateLine(QuadViewportKind),
    RotateLine(QuadViewportKind),
    AdjustSlab(QuadViewportKind),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuadReferenceSelection {
    pub view: QuadViewportKind,
    pub target: QuadReferenceTarget,
}

#[derive(Clone, Copy, Debug)]
pub enum QuadReferenceDrag {
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
    AdjustSlab {
        view: QuadViewportKind,
        line_kind: QuadViewportKind,
    },
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct FrameCacheKey {
    pub file_path: String,
    pub frame_index: u32,
    pub has_window_override: bool,
    pub window_center_bits: u64,
    pub window_width_bits: u64,
    pub lut_name: String,
    pub rotation_quarters: u8,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub invert: bool,
}

#[derive(Clone, Debug)]
pub struct CachedFrame {
    pub source_width: u32,
    pub source_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub rgba: Vec<u8>,
    pub window_center: f64,
    pub window_width: f64,
}

#[derive(Clone, Debug)]
pub enum DraftMeasurement {
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

#[derive(Clone, Debug)]
pub struct HandleDrag {
    pub measurement_id: String,
    pub handle_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewportGeometry {
    pub image_origin_x: f32,
    pub image_origin_y: f32,
    pub image_width: f32,
    pub image_height: f32,
}

/// Headless viewer session state — no UI-framework dependency.
pub struct ViewerState {
    pub imagebox: Rc<Imagebox>,
    pub series: Vec<SeriesInfo>,
    pub instances_by_series: HashMap<String, Vec<InstanceInfo>>,
    pub frames_by_series: HashMap<String, Vec<FrameRef>>,
    pub measurements_by_series: HashMap<String, Vec<Measurement>>,
    pub overlay_cache_by_file: HashMap<String, Vec<OverlayBitmap>>,
    pub active_series_uid: String,
    pub active_frame_index: usize,
    pub measurement_panel_visible: bool,
    pub volume_preview_active: bool,
    pub quad_viewport_active: bool,
    pub advanced_preview_mode: AdvancedPreviewMode,
    pub focused_quad_viewport: QuadViewportKind,
    pub active_tool: ViewerTool,
    pub viewport_scale: f32,
    pub viewport_offset_x: f32,
    pub viewport_offset_y: f32,
    pub window_center: Option<f64>,
    pub window_width: Option<f64>,
    pub default_window_center: Option<f64>,
    pub default_window_width: Option<f64>,
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub active_frame_width: u32,
    pub active_frame_height: u32,
    pub display_frame_width: u32,
    pub display_frame_height: u32,
    pub image_transform: ImageTransformState,
    pub active_lut_name: String,
    pub frame_cache: LruCache<FrameCacheKey, CachedFrame>,
    pub selected_measurement_id: Option<String>,
    pub draft_measurement: Option<DraftMeasurement>,
    pub handle_drag: Option<HandleDrag>,
    pub drag_state: Option<ViewportDragState>,
    pub volume_drag_state: Option<VolumeDragState>,
    pub volume_renderer: Option<VolumePreviewRenderer>,
    pub prepared_volumes_by_series: HashMap<String, PreparedVolume>,
    pub volume_view_state_by_series: HashMap<String, VolumeViewState>,
    pub slice_view_state_by_series: HashMap<String, SlicePreviewState>,
    pub quad_previews_by_kind: HashMap<QuadViewportKind, RgbaPreview>,
    pub quad_reference_lines_by_kind: HashMap<QuadViewportKind, Vec<QuadReferenceLineOverlay>>,
    pub quad_reference_hover: Option<QuadReferenceSelection>,
    pub quad_reference_drag: Option<QuadReferenceDrag>,
    pub pending_series_load_uid: Option<String>,
}

pub type ViewerSession = ViewerState;

impl ViewerState {
    pub fn new(imagebox: Rc<Imagebox>, series: Vec<SeriesInfo>, active_series_uid: String) -> Self {
        Self {
            imagebox,
            series,
            instances_by_series: HashMap::new(),
            frames_by_series: HashMap::new(),
            measurements_by_series: HashMap::new(),
            overlay_cache_by_file: HashMap::new(),
            active_series_uid,
            active_frame_index: 0,
            measurement_panel_visible: false,
            volume_preview_active: false,
            quad_viewport_active: false,
            advanced_preview_mode: AdvancedPreviewMode::default(),
            focused_quad_viewport: AdvancedPreviewMode::default().quad_viewport(),
            active_tool: ViewerTool::default(),
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
            pending_series_load_uid: None,
        }
    }
}

pub fn load_series_data(imagebox: &Imagebox, series_uid: &str) -> LeafResult<LoadedSeriesData> {
    let mut instances = imagebox.get_instances_for_series(&SeriesUid(series_uid.to_string()))?;
    sort_instances_for_stack(&mut instances);
    let frames = build_frames_for_series(&instances);
    let measurements = imagebox
        .load_measurements(series_uid)?
        .and_then(|json| serde_json::from_str::<Vec<Measurement>>(&json).ok())
        .unwrap_or_default();

    Ok(LoadedSeriesData {
        series_uid: series_uid.to_string(),
        instances,
        frames,
        measurements,
    })
}

pub fn apply_loaded_series_data(session: &mut ViewerSession, loaded: LoadedSeriesData) {
    let series_uid = loaded.series_uid.clone();
    session
        .instances_by_series
        .insert(series_uid.clone(), loaded.instances);
    session
        .frames_by_series
        .insert(series_uid.clone(), loaded.frames);
    if loaded.measurements.is_empty() {
        session.measurements_by_series.remove(&series_uid);
    } else {
        session
            .measurements_by_series
            .insert(series_uid, loaded.measurements);
    }
}

pub fn build_frames_for_series(instances: &[InstanceInfo]) -> Vec<FrameRef> {
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
    frames
}

#[derive(Clone, Debug)]
pub struct QuadReferenceLineOverlay {
    pub commands: String,
    pub start_x: f32,
    pub start_y: f32,
    pub end_x: f32,
    pub end_y: f32,
    pub handle1_x: f32,
    pub handle1_y: f32,
    pub handle2_x: f32,
    pub handle2_y: f32,
    pub handle3_x: f32,
    pub handle3_y: f32,
    pub handle4_x: f32,
    pub handle4_y: f32,
    pub source_kind: QuadViewportKind,
    pub slab_active: bool,
}

pub fn next_volume_blend_mode(mode: VolumeBlendMode) -> VolumeBlendMode {
    match mode {
        VolumeBlendMode::Composite => VolumeBlendMode::MaximumIntensity,
        VolumeBlendMode::MaximumIntensity => VolumeBlendMode::MinimumIntensity,
        VolumeBlendMode::MinimumIntensity => VolumeBlendMode::AverageIntensity,
        VolumeBlendMode::AverageIntensity => VolumeBlendMode::Composite,
    }
}

pub fn measurement_kind_label(measurement: &Measurement) -> &'static str {
    match &measurement.kind {
        MeasurementKind::Line { .. } => "Line",
        MeasurementKind::Angle { .. } => "Angle",
        MeasurementKind::RectangleRoi { .. } => "ROI",
        MeasurementKind::EllipseRoi { .. } => "Ellipse ROI",
        MeasurementKind::PolygonRoi { .. } => "Polygon ROI",
        MeasurementKind::PixelProbe { .. } => "Probe",
    }
}

pub fn measurement_overlay_text(
    measurement: &Measurement,
    pixel_spacing: (f64, f64),
    measurement_image: Option<&MeasurementImage<'_>>,
) -> String {
    match measurement_value(measurement, pixel_spacing, measurement_image) {
        MeasurementValue::Distance { mm } => format_distance_mm(mm),
        MeasurementValue::Angle { degrees } => format!("{degrees:.1}\u{b0}"),
        MeasurementValue::RoiStats {
            mean,
            area_mm2,
            pixel_count,
            ..
        } if pixel_count > 0 => format!("\u{3bc} {mean:.1} · {area_mm2:.1} mm\u{b2}"),
        MeasurementValue::RoiStats { area_mm2, .. } => format!("{area_mm2:.1} mm\u{b2}"),
        MeasurementValue::PixelValue { value, unit } => format_scalar_with_unit(value, &unit),
    }
}

pub fn measurement_panel_value_text(
    measurement: &Measurement,
    pixel_spacing: (f64, f64),
    measurement_image: Option<&MeasurementImage<'_>>,
) -> String {
    match measurement_value(measurement, pixel_spacing, measurement_image) {
        MeasurementValue::Distance { mm } => format_distance_mm(mm),
        MeasurementValue::Angle { degrees } => format!("{degrees:.1}\u{b0}"),
        MeasurementValue::RoiStats {
            mean,
            std_dev,
            min,
            max,
            area_mm2,
            pixel_count,
        } if pixel_count > 0 => format!(
            "\u{3bc} {mean:.1} \u{3c3} {std_dev:.1} [{min:.0}..{max:.0}] n={pixel_count} · {area_mm2:.1} mm\u{b2}"
        ),
        MeasurementValue::RoiStats { area_mm2, .. } => format!("{area_mm2:.1} mm\u{b2}"),
        MeasurementValue::PixelValue { value, unit } => format_scalar_with_unit(value, &unit),
    }
}

pub fn apply_overlays_to_rgba(
    rgba: &mut [u8],
    image_width: u32,
    image_height: u32,
    overlays: &[OverlayBitmap],
) {
    let image_width = image_width as usize;
    let image_height = image_height as usize;
    for overlay in overlays {
        let origin_y = overlay.origin.0.saturating_sub(1) as isize;
        let origin_x = overlay.origin.1.saturating_sub(1) as isize;
        let rows = overlay.rows as usize;
        let columns = overlay.columns as usize;
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

pub fn transformed_image_dimensions(
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

pub fn source_to_display_point_raw(
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

pub fn display_to_source_point_raw(
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

pub fn transform_rgba(
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

pub fn patient_orientation_label(vector: [f64; 3]) -> String {
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

pub fn orientation_labels_for_frame(
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

pub fn orientation_labels_for_slice_plane(plane: &SlicePlane) -> (String, String, String, String) {
    (
        patient_orientation_label([-plane.up.x, -plane.up.y, -plane.up.z]),
        patient_orientation_label([plane.up.x, plane.up.y, plane.up.z]),
        patient_orientation_label([-plane.right.x, -plane.right.y, -plane.right.z]),
        patient_orientation_label([plane.right.x, plane.right.y, plane.right.z]),
    )
}

pub fn mpr_uv_from_viewport(geometry: ViewportGeometry, x: f32, y: f32) -> Option<DVec2> {
    let normalized_x = (x - geometry.image_origin_x) / geometry.image_width;
    let normalized_y = (y - geometry.image_origin_y) / geometry.image_height;
    if !(0.0..=1.0).contains(&normalized_x) || !(0.0..=1.0).contains(&normalized_y) {
        return None;
    }
    Some(DVec2::new(normalized_x as f64, normalized_y as f64))
}

pub fn quad_world_to_viewport_point(
    plane: &SlicePlane,
    geometry: ViewportGeometry,
    world: DVec3,
) -> (f32, f32) {
    let (uv, _) = plane.world_to_point(world);
    (
        geometry.image_origin_x + uv.x as f32 * geometry.image_width,
        geometry.image_origin_y + uv.y as f32 * geometry.image_height,
    )
}

pub fn clip_line_to_geometry(
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

pub fn quad_reference_line_for_plane(
    current_plane: &SlicePlane,
    other_plane: &SlicePlane,
    other_state: &SlicePreviewState,
    source_kind: QuadViewportKind,
    shared_world: DVec3,
    geometry: ViewportGeometry,
) -> Option<QuadReferenceLineOverlay> {
    const SLAB_HANDLE_MIN_OFFSET_PX: f32 = 14.0;

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
    let line_dx = end_x - start_x;
    let line_dy = end_y - start_y;
    let line_length = (line_dx * line_dx + line_dy * line_dy).sqrt().max(1.0e-4);
    let slab_active = !matches!(other_state.projection_mode, SliceProjectionMode::Thin)
        && other_state.slab_half_thickness > 1.0e-4;
    let actual_slab_offset = if slab_active {
        let (slab_x, slab_y) = quad_world_to_viewport_point(
            current_plane,
            geometry,
            shared_world + other_plane.normal() * other_state.slab_half_thickness,
        );
        (slab_x - center_x, slab_y - center_y)
    } else {
        (0.0, 0.0)
    };
    let actual_slab_offset_len = (actual_slab_offset.0 * actual_slab_offset.0
        + actual_slab_offset.1 * actual_slab_offset.1)
        .sqrt();
    let slab_dir = if actual_slab_offset_len > 1.0e-4 {
        (
            actual_slab_offset.0 / actual_slab_offset_len,
            actual_slab_offset.1 / actual_slab_offset_len,
        )
    } else {
        (-line_dy / line_length, line_dx / line_length)
    };
    let slab_handle_offset_px = actual_slab_offset_len.max(SLAB_HANDLE_MIN_OFFSET_PX);
    let slab_handle_offset = (
        slab_dir.0 * slab_handle_offset_px,
        slab_dir.1 * slab_handle_offset_px,
    );
    let mut commands = vec![format!(
        "M {start_x:.1} {start_y:.1} L {end_x:.1} {end_y:.1}"
    )];
    if slab_active && actual_slab_offset_len > 0.5 {
        for sign in [-1.0f32, 1.0] {
            let offset_x = actual_slab_offset.0 * sign;
            let offset_y = actual_slab_offset.1 * sign;
            if let Some(((slab_start_x, slab_start_y), (slab_end_x, slab_end_y))) =
                clip_line_to_geometry(
                    center_x + offset_x,
                    center_y + offset_y,
                    dir_x - center_x,
                    dir_y - center_y,
                    geometry,
                )
            {
                commands.push(format!(
                    "M {slab_start_x:.1} {slab_start_y:.1} L {slab_end_x:.1} {slab_end_y:.1}"
                ));
            }
        }
    }
    Some(QuadReferenceLineOverlay {
        commands: commands.join(" "),
        start_x,
        start_y,
        end_x,
        end_y,
        handle1_x: (center_x + start_x) * 0.5,
        handle1_y: (center_y + start_y) * 0.5,
        handle2_x: (center_x + end_x) * 0.5,
        handle2_y: (center_y + end_y) * 0.5,
        handle3_x: center_x + slab_handle_offset.0,
        handle3_y: center_y + slab_handle_offset.1,
        handle4_x: center_x - slab_handle_offset.0,
        handle4_y: center_y - slab_handle_offset.1,
        source_kind,
        slab_active,
    })
}

pub fn normalized_angle_delta(current_angle_rad: f64, start_angle_rad: f64) -> f64 {
    let mut delta = current_angle_rad - start_angle_rad;
    while delta <= -std::f64::consts::PI {
        delta += std::f64::consts::TAU;
    }
    while delta > std::f64::consts::PI {
        delta -= std::f64::consts::TAU;
    }
    delta
}

pub fn point_to_segment_distance_sq(
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

pub fn displayed_image_geometry(
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

pub fn preview_dimensions_for_viewport(viewport_width: f32, viewport_height: f32) -> (u32, u32) {
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

pub fn preview_dimensions(state: &ViewerState) -> (u32, u32) {
    if state.volume_preview_active && !state.advanced_preview_mode.is_dvr() {
        if let Some(prepared) = state
            .prepared_volumes_by_series
            .get(&state.active_series_uid)
        {
            let slice_mode = state.advanced_preview_mode.slice_mode().unwrap_or_default();
            return slice_preview_dimensions(state, prepared, slice_mode);
        }
    }
    preview_dimensions_for_viewport(state.viewport_width, state.viewport_height)
}

pub fn slice_preview_dimensions(
    state: &ViewerState,
    prepared: &PreparedVolume,
    mode: SlicePreviewMode,
) -> (u32, u32) {
    slice_preview_dimensions_for_viewport(
        state.viewport_width,
        state.viewport_height,
        prepared,
        mode,
    )
}

pub fn slice_preview_dimensions_for_viewport(
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

pub fn fit_dimensions_to_aspect(max_width: u32, max_height: u32, aspect_ratio: f64) -> (u32, u32) {
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

pub fn sort_instances_for_stack(instances: &mut [InstanceInfo]) {
    if instances.len() <= 1 {
        return;
    }

    hydrate_instance_geometry(instances);

    let mut reference_candidates = instances.iter().collect::<Vec<_>>();
    reference_candidates.sort_by(|a, b| compare_instances_by_fallback(a, b));
    let Some(reference) = reference_candidates.first() else {
        return;
    };
    let (Some(reference_ipp), Some(reference_iop)) = (
        reference.image_position_patient,
        reference.image_orientation_patient,
    ) else {
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

pub fn to_rgba(frame: &DecodedFrame, lut_name: &str) -> LeafResult<Vec<u8>> {
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

fn measurement_value(
    measurement: &Measurement,
    pixel_spacing: (f64, f64),
    measurement_image: Option<&MeasurementImage<'_>>,
) -> MeasurementValue {
    measurement
        .compute_with_image(pixel_spacing, measurement_image)
        .value
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

fn format_scalar_with_unit(value: f64, unit: &str) -> String {
    if unit.is_empty() {
        format!("{value:.0}")
    } else {
        format!("{value:.0} {unit}")
    }
}

fn resolve_color_lut(name: &str) -> ColorLut {
    match name {
        "grayscale_inverted" => ColorLut::grayscale_inverted(),
        "hot_iron" => ColorLut::hot_iron(),
        "bone" => ColorLut::bone(),
        _ => ColorLut::grayscale(),
    }
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

pub fn active_volume_view_state(state: &mut ViewerState) -> &mut VolumeViewState {
    let series_uid = state.active_series_uid.clone();
    state
        .volume_view_state_by_series
        .entry(series_uid)
        .or_default()
}

pub fn active_slice_view_state(state: &mut ViewerState) -> &mut SlicePreviewState {
    let series_uid = state.active_series_uid.clone();
    state
        .slice_view_state_by_series
        .entry(series_uid)
        .or_default()
}

pub fn layout_label(state: &ViewerState) -> &'static str {
    if !state.volume_preview_active {
        "1Up"
    } else if state.quad_viewport_active {
        "Quad"
    } else if state.advanced_preview_mode.is_dvr() {
        "DVR"
    } else {
        "MPR"
    }
}

pub fn set_selected_quad_viewport(state: &mut ViewerState, kind: QuadViewportKind) {
    state.focused_quad_viewport = kind;
    state.advanced_preview_mode = kind.advanced_preview_mode();
}

pub fn quad_preview(state: &ViewerState, kind: QuadViewportKind) -> Option<&RgbaPreview> {
    state.quad_previews_by_kind.get(&kind)
}

pub fn quad_preview_info(state: &ViewerState, kind: QuadViewportKind) -> String {
    quad_preview(state, kind)
        .map(|preview| preview.info.clone())
        .unwrap_or_default()
}

pub fn quad_tile_max_dimensions(state: &ViewerState) -> (f32, f32) {
    let width = if state.viewport_width > 0.0 {
        state.viewport_width * 0.5
    } else {
        512.0
    };
    let height = if state.viewport_height > 0.0 {
        state.viewport_height * 0.5
    } else {
        512.0
    };
    (width.max(256.0), height.max(256.0))
}

pub fn quad_mpr_preview_info(kind: QuadViewportKind, state: SlicePreviewState) -> String {
    format!(
        "{} {}",
        kind.title(),
        slice_projection_mode_label(state.projection_mode)
    )
}

pub fn slice_projection_mode_label(mode: SliceProjectionMode) -> &'static str {
    match mode {
        SliceProjectionMode::Thin => "Thin",
        SliceProjectionMode::MaximumIntensity => "MIP",
        SliceProjectionMode::MinimumIntensity => "MinIP",
        SliceProjectionMode::AverageIntensity => "Avg",
    }
}

pub fn volume_blend_mode_label(mode: VolumeBlendMode) -> &'static str {
    match mode {
        VolumeBlendMode::Composite => "Comp",
        VolumeBlendMode::MaximumIntensity => "MIP",
        VolumeBlendMode::MinimumIntensity => "MinIP",
        VolumeBlendMode::AverageIntensity => "Avg",
    }
}

pub fn quad_viewport_geometry(
    state: &ViewerState,
    kind: QuadViewportKind,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<ViewportGeometry> {
    let preview = quad_preview(state, kind)?;
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

pub fn quad_mpr_angle_from_viewport(
    state: &ViewerState,
    kind: QuadViewportKind,
    x: f32,
    y: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<f64> {
    let prepared = state
        .prepared_volumes_by_series
        .get(&state.active_series_uid)?;
    let slice_state = quad_slice_view_state_for_kind(state, prepared, kind)?;
    let bounds = prepared.world_bounds();
    let plane = slice_state.slice_plane(bounds);
    let shared_world = slice_state.crosshair_world(bounds);
    let pointer_world =
        quad_mpr_world_point_from_viewport(state, kind, x, y, viewport_width, viewport_height)?;
    let offset = pointer_world - shared_world;
    let dx = offset.dot(plane.right);
    let dy = offset.dot(plane.up);
    if dx.abs() < 1.0e-4_f64 && dy.abs() < 1.0e-4_f64 {
        return None;
    }
    Some(dy.atan2(dx))
}

pub fn quad_crosshair_viewport_point(
    state: &ViewerState,
    kind: QuadViewportKind,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<(f32, f32)> {
    let geometry = quad_viewport_geometry(state, kind, viewport_width, viewport_height)?;
    let prepared = state
        .prepared_volumes_by_series
        .get(&state.active_series_uid)?;
    let slice_state = quad_slice_view_state_for_kind(state, prepared, kind)?;
    let bounds = prepared.world_bounds();
    let plane = slice_state.slice_plane(bounds);
    let shared_world = slice_state.crosshair_world(bounds);
    Some(quad_world_to_viewport_point(&plane, geometry, shared_world))
}

pub fn quad_reference_line_hit(
    state: &ViewerState,
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
        quad_crosshair_viewport_point(state, kind, viewport_width, viewport_height)
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

    if let Some((_, source_kind)) = state
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

    if let Some((_, source_kind)) = state
        .quad_reference_lines_by_kind
        .get(&kind)
        .into_iter()
        .flat_map(|lines| lines.iter())
        .filter_map(|line| {
            let distance_sq = point_to_segment_distance_sq(
                x,
                y,
                line.handle3_x,
                line.handle3_y,
                line.handle3_x,
                line.handle3_y,
            )
            .min(point_to_segment_distance_sq(
                x,
                y,
                line.handle4_x,
                line.handle4_y,
                line.handle4_x,
                line.handle4_y,
            ));
            (distance_sq <= HANDLE_THRESHOLD_SQ).then_some((distance_sq, line.source_kind))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
    {
        return Some(QuadReferenceSelection {
            view: kind,
            target: QuadReferenceTarget::AdjustSlab(source_kind),
        });
    }

    state
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
            let slab_handle_distance_sq = point_to_segment_distance_sq(
                x,
                y,
                line.handle3_x,
                line.handle3_y,
                line.handle3_x,
                line.handle3_y,
            )
            .min(point_to_segment_distance_sq(
                x,
                y,
                line.handle4_x,
                line.handle4_y,
                line.handle4_x,
                line.handle4_y,
            ));
            (line_distance_sq <= LINE_THRESHOLD_SQ
                && handle_distance_sq > HANDLE_THRESHOLD_SQ
                && slab_handle_distance_sq > HANDLE_THRESHOLD_SQ)
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

pub fn quad_reference_line_highlighted(
    state: &ViewerState,
    view: QuadViewportKind,
    source_kind: QuadViewportKind,
) -> bool {
    let drag_active = matches!(
        state.quad_reference_drag,
        Some(QuadReferenceDrag::TranslateLine {
            view: drag_view,
            line_kind,
            ..
        } | QuadReferenceDrag::RotateLine {
            view: drag_view,
            line_kind,
            ..
        } | QuadReferenceDrag::AdjustSlab {
            view: drag_view,
            line_kind,
            ..
        }) if (drag_view, line_kind) == (view, source_kind)
    );
    let hover_active = matches!(
        state.quad_reference_hover,
        Some(QuadReferenceSelection {
            view: hover_view,
            target: QuadReferenceTarget::TranslateLine(line_kind)
                | QuadReferenceTarget::RotateLine(line_kind)
                | QuadReferenceTarget::AdjustSlab(line_kind),
        }) if (hover_view, line_kind) == (view, source_kind)
    );
    drag_active || hover_active
}

pub fn quad_reference_center_highlighted(state: &ViewerState, view: QuadViewportKind) -> bool {
    matches!(
        state.quad_reference_drag,
        Some(QuadReferenceDrag::Center { view: drag_view }) if drag_view == view
    ) || matches!(
        state.quad_reference_hover,
        Some(QuadReferenceSelection {
            view: hover_view,
            target: QuadReferenceTarget::Center,
        }) if hover_view == view
    )
}

pub fn apply_volume_drag(
    state: &mut ViewerState,
    start_view_state: VolumeViewState,
    delta_x: f32,
    delta_y: f32,
    button: i32,
) {
    let active_tool = state.active_tool;
    let scalar_range = state
        .prepared_volumes_by_series
        .get(&state.active_series_uid)
        .map(|prepared| prepared.scalar_range());
    let view_state = active_volume_view_state(state);
    *view_state = start_view_state;
    if let Some((scalar_min, scalar_max)) = scalar_range {
        view_state.ensure_transfer_window(scalar_min, scalar_max);
    }

    if button == 2 {
        view_state.orbit(delta_x as f64 * 0.6, -delta_y as f64 * 0.6);
        return;
    }

    match active_tool {
        ViewerTool::WindowLevel => {
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
        ViewerTool::Pan => view_state.pan(-delta_x as f64, -delta_y as f64),
        ViewerTool::Zoom => {
            let factor = (1.0 - delta_y as f64 * 0.004).clamp(0.1, 10.0);
            view_state.zoom_by(factor);
        }
        _ => view_state.orbit(delta_x as f64 * 0.6, -delta_y as f64 * 0.6),
    }
}

pub fn update_viewport_dimensions(
    state: &mut ViewerState,
    viewport_width: f32,
    viewport_height: f32,
) {
    if viewport_width > 0.0 {
        state.viewport_width = viewport_width;
    }
    if viewport_height > 0.0 {
        state.viewport_height = viewport_height;
    }
}

pub fn active_pixel_spacing(state: &ViewerState) -> (f64, f64) {
    state
        .series
        .iter()
        .find(|series| series.series_uid.0 == state.active_series_uid)
        .and_then(|series| series.pixel_spacing)
        .unwrap_or((1.0, 1.0))
}

pub fn measurement_frame_for_slice(
    state: &ViewerState,
    slice_index: usize,
) -> Option<MeasurementFrame> {
    let frame_ref = state
        .frames_by_series
        .get(&state.active_series_uid)?
        .get(slice_index)?;
    match decode_frame_for_measurements(Path::new(&frame_ref.file_path), frame_ref.frame_index) {
        Ok(frame) => Some(frame),
        Err(error) => {
            info!(
                "Failed to decode source pixels for measurement statistics on slice {}: {}",
                slice_index + 1,
                error
            );
            None
        }
    }
}

pub fn active_measurement_frame(state: &ViewerState) -> Option<MeasurementFrame> {
    measurement_frame_for_slice(state, state.active_frame_index)
}

pub fn measurement_image_view<'a>(frame: &'a MeasurementFrame) -> MeasurementImage<'a> {
    MeasurementImage {
        width: frame.width,
        height: frame.height,
        pixels: &frame.pixels,
        unit: &frame.unit,
    }
}

pub fn frame_cache_key(state: &ViewerState, frame_ref: &FrameRef) -> FrameCacheKey {
    let (has_window_override, window_center_bits, window_width_bits) =
        match state.window_center.zip(state.window_width) {
            Some((center, width)) => (true, center.to_bits(), width.to_bits()),
            None => (false, 0, 0),
        };
    FrameCacheKey {
        file_path: frame_ref.file_path.clone(),
        frame_index: frame_ref.frame_index,
        has_window_override,
        window_center_bits,
        window_width_bits,
        lut_name: state.active_lut_name.clone(),
        rotation_quarters: state.image_transform.rotation_quarters % 4,
        flip_horizontal: state.image_transform.flip_horizontal,
        flip_vertical: state.image_transform.flip_vertical,
        invert: state.image_transform.invert,
    }
}

pub fn image_to_viewport_point(state: &ViewerState, point: DVec2) -> Option<(f32, f32)> {
    let geometry = current_viewport_geometry(state)?;
    let source_width = state.active_frame_width as f64;
    let source_height = state.active_frame_height as f64;
    let display_width = state.display_frame_width as f32;
    let display_height = state.display_frame_height as f32;
    if source_width <= 0.0 || source_height <= 0.0 || display_width <= 0.0 || display_height <= 0.0
    {
        return None;
    }

    let display_point =
        source_to_display_point_raw(point, source_width, source_height, state.image_transform);
    Some((
        geometry.image_origin_x + (display_point.x as f32 / display_width) * geometry.image_width,
        geometry.image_origin_y + (display_point.y as f32 / display_height) * geometry.image_height,
    ))
}

pub fn viewport_to_image_point(state: &ViewerState, x: f32, y: f32, clamp: bool) -> Option<DVec2> {
    let geometry = current_viewport_geometry(state)?;
    let source_width = state.active_frame_width as f64;
    let source_height = state.active_frame_height as f64;
    let display_width = state.display_frame_width as f32;
    let display_height = state.display_frame_height as f32;
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
        state.image_transform,
    );

    Some(DVec2::new(
        source_point.x.clamp(0.0, source_width),
        source_point.y.clamp(0.0, source_height),
    ))
}

pub fn current_viewport_geometry(state: &ViewerState) -> Option<ViewportGeometry> {
    displayed_image_geometry(
        state.viewport_width,
        state.viewport_height,
        state.display_frame_width,
        state.display_frame_height,
        state.viewport_scale,
        state.viewport_offset_x,
        state.viewport_offset_y,
    )
}

pub fn mpr_world_point_from_viewport(state: &ViewerState, x: f32, y: f32) -> Option<DVec3> {
    if !state.volume_preview_active || state.advanced_preview_mode.is_dvr() {
        return None;
    }
    let geometry = current_viewport_geometry(state)?;
    let prepared = state
        .prepared_volumes_by_series
        .get(&state.active_series_uid)?;
    let slice_state = state
        .slice_view_state_by_series
        .get(&state.active_series_uid)
        .copied()
        .unwrap_or_default();
    let slice_uv = mpr_uv_from_viewport(geometry, x, y)?;
    let plane = slice_state.slice_plane(prepared.world_bounds());
    Some(plane.point_to_world(slice_uv))
}

pub fn quad_slice_view_state_for_kind(
    state: &ViewerState,
    prepared: &PreparedVolume,
    kind: QuadViewportKind,
) -> Option<SlicePreviewState> {
    let slice_mode = kind.slice_mode()?;
    let scalar_range = prepared.scalar_range();
    let bounds = prepared.world_bounds();
    let mut slice_state = state
        .slice_view_state_by_series
        .get(&state.active_series_uid)
        .copied()
        .unwrap_or_default();
    if slice_state.mode != slice_mode {
        slice_state.set_mode(slice_mode);
    }
    slice_state.ensure_transfer_window(scalar_range.0, scalar_range.1);
    slice_state.center_on_crosshair(bounds);
    Some(slice_state)
}

pub fn quad_mpr_world_point_from_viewport(
    state: &ViewerState,
    kind: QuadViewportKind,
    x: f32,
    y: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<DVec3> {
    let preview = quad_preview(state, kind)?;
    let geometry = displayed_image_geometry(
        viewport_width,
        viewport_height,
        preview.width,
        preview.height,
        1.0,
        0.0,
        0.0,
    )?;
    let prepared = state
        .prepared_volumes_by_series
        .get(&state.active_series_uid)?;
    let slice_state = quad_slice_view_state_for_kind(state, prepared, kind)?;
    let slice_uv = mpr_uv_from_viewport(geometry, x, y)?;
    let plane = slice_state.slice_plane(prepared.world_bounds());
    Some(plane.point_to_world(slice_uv))
}

pub fn adjust_quad_crosshair_by_scroll(
    state: &mut ViewerState,
    kind: QuadViewportKind,
    delta: f32,
) -> LeafResult<()> {
    let Some(slice_mode) = kind.slice_mode() else {
        return Ok(());
    };
    let prepared = state
        .prepared_volumes_by_series
        .get(&state.active_series_uid)
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
    let slice_state = active_slice_view_state(state);
    if slice_state.mode != slice_mode {
        slice_state.set_mode(slice_mode);
    }
    let normal = slice_state.slice_plane(bounds).normal();
    let world = slice_state.crosshair_world(bounds) + normal * delta_mm;
    slice_state.center_on_world(world, bounds);
    Ok(())
}

pub fn find_handle_at(state: &ViewerState, x: f32, y: f32) -> Option<(String, usize)> {
    let threshold_sq = 64.0f32;
    let measurements = state.measurements_by_series.get(&state.active_series_uid)?;
    for measurement in measurements
        .iter()
        .filter(|m| m.slice_index == state.active_frame_index)
    {
        let handles = measurement.handle_positions();
        for (idx, handle) in handles.iter().enumerate() {
            if let Some((hx, hy)) = image_to_viewport_point(state, *handle) {
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

pub fn persist_measurements(state: &ViewerState) {
    let series_uid = &state.active_series_uid;
    if let Some(measurements) = state.measurements_by_series.get(series_uid) {
        if let Ok(json) = serde_json::to_string(measurements) {
            if let Err(e) = state.imagebox.store_measurements(series_uid, &json) {
                info!("Failed to persist measurements: {}", e);
            }
        }
    } else {
        let _ = state.imagebox.delete_measurements(series_uid);
    }
}

pub fn reset_viewport_state(state: &mut ViewerState, clear_defaults: bool) {
    state.viewport_scale = 1.0;
    state.viewport_offset_x = 0.0;
    state.viewport_offset_y = 0.0;
    state.image_transform = ImageTransformState::default();
    state.active_lut_name = DEFAULT_LUT_NAME.to_string();
    state.drag_state = None;
    state.volume_drag_state = None;

    if clear_defaults {
        state.window_center = None;
        state.window_width = None;
        state.default_window_center = None;
        state.default_window_width = None;
    } else {
        state.window_center = state.default_window_center;
        state.window_width = state.default_window_width;
    }
}

pub fn active_series_file_paths(state: &ViewerState) -> LeafResult<Vec<String>> {
    let instances = state
        .instances_by_series
        .get(&state.active_series_uid)
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

pub fn ensure_active_prepared_volume(state: &mut ViewerState) -> LeafResult<String> {
    let series_uid = state.active_series_uid.clone();
    if !state.prepared_volumes_by_series.contains_key(&series_uid) {
        let file_paths = active_series_file_paths(state)?;
        let prepared = {
            let renderer = ensure_volume_renderer(state)?;
            renderer.prepare_series_volume(&file_paths, &SeriesUid(series_uid.clone()))?
        };
        state
            .prepared_volumes_by_series
            .insert(series_uid.clone(), prepared);
    }
    Ok(series_uid)
}

pub fn ensure_volume_renderer(state: &mut ViewerState) -> LeafResult<&mut VolumePreviewRenderer> {
    if state.volume_renderer.is_none() {
        state.volume_renderer = Some(VolumePreviewRenderer::new()?);
    }
    state
        .volume_renderer
        .as_mut()
        .ok_or_else(|| LeafError::Render("Volume renderer unavailable".into()))
}

pub fn lut_label(name: &str) -> &'static str {
    LUT_PRESETS
        .iter()
        .find_map(|(label, lut_name)| (*lut_name == name).then_some(*label))
        .unwrap_or("Gray")
}

pub fn next_lut_name(current: &str) -> &'static str {
    LUT_PRESETS
        .iter()
        .position(|(_, lut_name)| *lut_name == current)
        .map(|index| LUT_PRESETS[(index + 1) % LUT_PRESETS.len()].1)
        .unwrap_or(DEFAULT_LUT_NAME)
}

pub fn render_volume_image(
    state: &mut ViewerState,
    interactive: bool,
) -> LeafResult<VolumePreviewImage> {
    let series_uid = ensure_active_prepared_volume(state)?;
    ensure_volume_renderer(state)?;
    let preview_size = preview_dimensions(state);
    let scalar_range = state
        .prepared_volumes_by_series
        .get(&series_uid)
        .map(|prepared| prepared.scalar_range())
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    active_volume_view_state(state).ensure_transfer_window(scalar_range.0, scalar_range.1);
    let view_state = state
        .volume_view_state_by_series
        .get(&series_uid)
        .copied()
        .unwrap_or_default();
    let prepared = state
        .prepared_volumes_by_series
        .get(&series_uid)
        .cloned()
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    let renderer = state
        .volume_renderer
        .as_mut()
        .ok_or_else(|| LeafError::Render("Volume renderer unavailable".into()))?;
    renderer.render_prepared_preview(
        &prepared,
        &view_state,
        preview_size.0,
        preview_size.1,
        interactive,
    )
}

pub fn render_slice_image(state: &mut ViewerState) -> LeafResult<VolumePreviewImage> {
    let series_uid = ensure_active_prepared_volume(state)?;
    ensure_volume_renderer(state)?;
    let prepared = state
        .prepared_volumes_by_series
        .get(&series_uid)
        .cloned()
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    let scalar_range = prepared.scalar_range();
    let bounds = prepared.world_bounds();
    let slice_mode = state.advanced_preview_mode.slice_mode().unwrap_or_default();
    let preview_size = slice_preview_dimensions(state, &prepared, slice_mode);
    {
        let slice_state = active_slice_view_state(state);
        if slice_state.mode != slice_mode {
            slice_state.set_mode(slice_mode);
        }
        slice_state.ensure_transfer_window(scalar_range.0, scalar_range.1);
        slice_state.center_on_crosshair(bounds);
    }
    let view_state = state
        .slice_view_state_by_series
        .get(&series_uid)
        .copied()
        .unwrap_or_else(|| {
            let mut state = SlicePreviewState::default();
            state.set_mode(slice_mode);
            state
        });
    let renderer = state
        .volume_renderer
        .as_mut()
        .ok_or_else(|| LeafError::Render("Volume renderer unavailable".into()))?;
    renderer.render_prepared_slice_preview(
        &prepared,
        &view_state,
        preview_size.0,
        preview_size.1,
        true,
    )
}

pub fn render_quad_rgba(
    state: &mut ViewerState,
    kind: QuadViewportKind,
    interactive: bool,
) -> LeafResult<RgbaPreview> {
    let series_uid = ensure_active_prepared_volume(state)?;
    ensure_volume_renderer(state)?;
    let prepared = state
        .prepared_volumes_by_series
        .get(&series_uid)
        .cloned()
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    let (tile_width, tile_height) = quad_tile_max_dimensions(state);

    if kind.is_dvr() {
        let preview_size = preview_dimensions_for_viewport(tile_width, tile_height);
        let scalar_range = prepared.scalar_range();
        active_volume_view_state(state).ensure_transfer_window(scalar_range.0, scalar_range.1);
        let view_state = state
            .volume_view_state_by_series
            .get(&series_uid)
            .copied()
            .unwrap_or_default();
        let renderer = state
            .volume_renderer
            .as_mut()
            .ok_or_else(|| LeafError::Render("Volume renderer unavailable".into()))?;
        let preview = renderer.render_prepared_preview(
            &prepared,
            &view_state,
            preview_size.0,
            preview_size.1,
            interactive,
        )?;
        return Ok(RgbaPreview {
            width: preview.width,
            height: preview.height,
            rgba: preview.rgba,
            info: format!("DVR {}", volume_blend_mode_label(view_state.blend_mode)),
        });
    }

    let slice_state = quad_slice_view_state_for_kind(state, &prepared, kind)
        .ok_or_else(|| LeafError::Render("Quad MPR state unavailable".into()))?;
    let preview_size = slice_preview_dimensions_for_viewport(
        tile_width,
        tile_height,
        &prepared,
        kind.slice_mode().unwrap_or_default(),
    );
    let renderer = state
        .volume_renderer
        .as_mut()
        .ok_or_else(|| LeafError::Render("Volume renderer unavailable".into()))?;
    let preview = renderer.render_prepared_slice_preview(
        &prepared,
        &slice_state,
        preview_size.0,
        preview_size.1,
        false,
    )?;
    Ok(RgbaPreview {
        width: preview.width,
        height: preview.height,
        rgba: preview.rgba,
        info: quad_mpr_preview_info(kind, slice_state),
    })
}

pub fn build_quad_reference_lines(state: &mut ViewerState) -> LeafResult<()> {
    if !state.volume_preview_active || !state.quad_viewport_active {
        state.quad_reference_lines_by_kind.clear();
        return Ok(());
    }

    let prepared = state
        .prepared_volumes_by_series
        .get(&state.active_series_uid)
        .cloned()
        .ok_or_else(|| LeafError::Render("Prepared volume missing".into()))?;
    let (tile_width, tile_height) = quad_tile_max_dimensions(state);
    for kind in [
        QuadViewportKind::Axial,
        QuadViewportKind::Coronal,
        QuadViewportKind::Sagittal,
    ] {
        let Some(current_state) = quad_slice_view_state_for_kind(state, &prepared, kind) else {
            continue;
        };
        let Some(geometry) = quad_viewport_geometry(state, kind, tile_width, tile_height) else {
            continue;
        };
        let current_plane = current_state.slice_plane(prepared.world_bounds());
        let shared_world = current_state.crosshair_world(prepared.world_bounds());
        let overlays = kind
            .linked_mpr_views()
            .into_iter()
            .filter_map(|other_kind| {
                quad_slice_view_state_for_kind(state, &prepared, other_kind).and_then(
                    |other_state| {
                        quad_reference_line_for_plane(
                            &current_plane,
                            &other_state.slice_plane(prepared.world_bounds()),
                            &other_state,
                            other_kind,
                            shared_world,
                            geometry,
                        )
                    },
                )
            })
            .collect::<Vec<_>>();
        state.quad_reference_lines_by_kind.insert(kind, overlays);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use leaf_core::domain::{SopInstanceUid, StudyUid};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn test_imagebox() -> Rc<Imagebox> {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/leaf-viewer-tests");
        std::fs::create_dir_all(&base).expect("create test artifact dir");
        let id = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let path = base.join(format!("viewer-{id}.redb"));
        let _ = std::fs::remove_file(&path);
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
        let mut session = ViewerSession::new(test_imagebox(), Vec::new(), String::new());
        session.viewport_width = 1400.0;
        session.viewport_height = 900.0;

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

        assert_eq!(
            measurement_overlay_text(&measurement, (1.0, 1.0), None),
            "5.00 mm"
        );
    }

    #[test]
    fn roi_panel_values_include_statistics_when_pixels_are_available() {
        let measurement =
            Measurement::rectangle_roi("series", 0, DVec2::new(1.0, 1.0), DVec2::new(3.0, 3.0));
        let pixels = vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0, //
            12.0, 13.0, 14.0, 15.0,
        ];
        let image = MeasurementImage {
            width: 4,
            height: 4,
            pixels: &pixels,
            unit: "HU",
        };

        assert_eq!(
            measurement_panel_value_text(&measurement, (1.0, 1.0), Some(&image)),
            "\u{3bc} 7.5 \u{3c3} 2.1 [5..10] n=4 · 4.0 mm\u{b2}"
        );
    }

    #[test]
    fn grayscale_lut_maps_pixels_to_false_color() {
        let frame = DecodedFrame {
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
        let mut other_state = SlicePreviewState::default();
        other_state.set_mode(SlicePreviewMode::Coronal);
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
            &other_state,
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
    fn thick_slab_reference_lines_add_perpendicular_handles_and_guides() {
        let plane = SlicePlane::new(DVec3::ZERO, DVec3::X, DVec3::Y, 10.0, 10.0);
        let other_plane = SlicePlane::new(DVec3::ZERO, DVec3::Y, DVec3::Z, 10.0, 10.0);
        let mut other_state = SlicePreviewState::default();
        other_state.set_mode(SlicePreviewMode::Coronal);
        other_state.set_slab_half_thickness_from_drag(
            2.0,
            0.5,
            SliceProjectionMode::MaximumIntensity,
        );
        let geometry = ViewportGeometry {
            image_origin_x: 0.0,
            image_origin_y: 0.0,
            image_width: 100.0,
            image_height: 100.0,
        };

        let overlay = quad_reference_line_for_plane(
            &plane,
            &other_plane,
            &other_state,
            QuadViewportKind::Coronal,
            DVec3::ZERO,
            geometry,
        )
        .expect("reference line should exist");

        assert!(overlay.slab_active);
        assert_eq!(overlay.commands.matches('M').count(), 3);
        assert!(
            point_to_segment_distance_sq(
                overlay.handle3_x,
                overlay.handle3_y,
                overlay.start_x,
                overlay.start_y,
                overlay.end_x,
                overlay.end_y,
            ) > 1.0
        );
        assert!(
            point_to_segment_distance_sq(
                overlay.handle4_x,
                overlay.handle4_y,
                overlay.start_x,
                overlay.start_y,
                overlay.end_x,
                overlay.end_y,
            ) > 1.0
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
