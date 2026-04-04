//! Local viewer state and rendering flow for pacsleaf.

use crate::browser::{
    apply_window_geometry, capture_window_geometry, load_window_geometry, save_window_geometry,
    VIEWER_WINDOW_GEOMETRY_KEY,
};
use glam::DVec2;
use leaf_core::domain::StudyUid;
use leaf_core::error::{LeafError, LeafResult};
use leaf_db::imagebox::Imagebox;
use leaf_dicom::overlay::{load_overlays, OverlayBitmap};
use leaf_dicom::pixel::{
    decode_frame_for_measurements, decode_frame_with_window, MeasurementFrame,
};
use leaf_render::{
    PreparedVolume, SlicePlane, SlicePreviewState, SliceProjectionMode, VolumeBlendMode,
    VolumePreviewImage, VolumeViewState,
};
use leaf_tools::measurement::{Measurement, MeasurementImage, MeasurementKind};
use slint::{ComponentHandle, ModelRc, Timer, TimerMode, VecModel};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;
use tracing::info;

const THUMB_SIZE: usize = leaf_viewer::THUMB_SIZE;
const DEFAULT_LUT_NAME: &str = leaf_viewer::DEFAULT_LUT_NAME;
const LUT_PRESETS: [(&str, &str); 4] = leaf_viewer::LUT_PRESETS;

type FrameRef = leaf_viewer::FrameRef;
type ViewerState = leaf_viewer::ViewerState;
type ViewerTool = leaf_viewer::ViewerTool;
type LoadedSeriesData = leaf_viewer::LoadedSeriesData;
type ViewportDragState = leaf_viewer::ViewportDragState;
type VolumeDragState = leaf_viewer::VolumeDragState;
type ImageTransformState = leaf_viewer::ImageTransformState;
type AdvancedPreviewMode = leaf_viewer::AdvancedPreviewMode;
type QuadViewportKind = leaf_viewer::QuadViewportKind;
type QuadReferenceTarget = leaf_viewer::QuadReferenceTarget;
type QuadReferenceSelection = leaf_viewer::QuadReferenceSelection;
type QuadReferenceDrag = leaf_viewer::QuadReferenceDrag;
type FrameCacheKey = leaf_viewer::FrameCacheKey;
type CachedFrame = leaf_viewer::CachedFrame;
type DraftMeasurement = leaf_viewer::DraftMeasurement;
type HandleDrag = leaf_viewer::HandleDrag;
type ViewportGeometry = leaf_viewer::ViewportGeometry;

#[derive(Clone)]
struct AdvancedViewportPreview {
    image: slint::Image,
    info: String,
}

struct ViewerSession {
    state: ViewerState,
    viewer: slint::Weak<leaf_ui::StudyViewerWindow>,
    thumbnails_by_series: HashMap<String, slint::Image>,
    series_load_timer: Option<Timer>,
}

impl Deref for ViewerSession {
    type Target = ViewerState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl DerefMut for ViewerSession {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

fn shared_viewer_tool(tool: leaf_ui::ViewerTool) -> ViewerTool {
    match tool {
        leaf_ui::ViewerTool::WindowLevel => ViewerTool::WindowLevel,
        leaf_ui::ViewerTool::Pan => ViewerTool::Pan,
        leaf_ui::ViewerTool::Zoom => ViewerTool::Zoom,
        leaf_ui::ViewerTool::Scroll => ViewerTool::Scroll,
        leaf_ui::ViewerTool::Line => ViewerTool::Line,
        leaf_ui::ViewerTool::Angle => ViewerTool::Angle,
        leaf_ui::ViewerTool::RectangleRoi => ViewerTool::RectangleRoi,
        leaf_ui::ViewerTool::EllipseRoi => ViewerTool::EllipseRoi,
        leaf_ui::ViewerTool::Annotation => ViewerTool::Annotation,
    }
}

fn ui_viewer_tool(tool: ViewerTool) -> leaf_ui::ViewerTool {
    match tool {
        ViewerTool::WindowLevel => leaf_ui::ViewerTool::WindowLevel,
        ViewerTool::Pan => leaf_ui::ViewerTool::Pan,
        ViewerTool::Zoom => leaf_ui::ViewerTool::Zoom,
        ViewerTool::Scroll => leaf_ui::ViewerTool::Scroll,
        ViewerTool::Line => leaf_ui::ViewerTool::Line,
        ViewerTool::Angle => leaf_ui::ViewerTool::Angle,
        ViewerTool::RectangleRoi => leaf_ui::ViewerTool::RectangleRoi,
        ViewerTool::EllipseRoi => leaf_ui::ViewerTool::EllipseRoi,
        ViewerTool::Annotation => leaf_ui::ViewerTool::Annotation,
    }
}

fn to_shared_transform(transform: ImageTransformState) -> leaf_viewer::ImageTransformState {
    transform
}

fn to_shared_viewport_geometry(geometry: ViewportGeometry) -> leaf_viewer::ViewportGeometry {
    geometry
}

fn from_shared_viewport_geometry(geometry: leaf_viewer::ViewportGeometry) -> ViewportGeometry {
    geometry
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

    // Load only cached thumbnails during startup; avoid decoding series data
    // before the viewer is shown.
    let mut thumbnails_by_series = HashMap::new();
    for series_info in &series {
        let uid = &series_info.series_uid.0;
        if let Ok(Some(rgba)) = imagebox.load_thumbnail(uid) {
            if rgba.len() == THUMB_SIZE * THUMB_SIZE * 4 {
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
    if active_series_uid.is_empty() {
        viewer.set_connection_status("Local imagebox".into());
        viewer.set_slice_info("0/0".into());
    } else {
        viewer.set_connection_status("Loading series...".into());
        viewer.set_slice_info("Loading...".into());
        viewer.set_loading_visible(true);
        viewer.set_loading_title("Opening viewer".into());
        viewer.set_loading_message("Loading the first series...".into());
        viewer.set_loading_current(0);
        viewer.set_loading_total(0);
    }
    let session = Rc::new(RefCell::new(ViewerSession {
        state: ViewerState::new(imagebox.clone(), series, active_series_uid),
        viewer: viewer.as_weak(),
        thumbnails_by_series,
        series_load_timer: None,
    }));

    {
        let session_ref = session.borrow();
        update_series_model(&session_ref)?;
        update_measurements_model(&session_ref)?;
        let _ = update_measurement_overlays(&session_ref);
        apply_viewport_state(&session_ref)?;
    }

    let session_for_series = session.clone();
    viewer.on_series_selected(move |series_uid| {
        let selected_series_uid = series_uid.to_string();
        let should_load = {
            let mut session = session_for_series.borrow_mut();
            let preview_active = session.volume_preview_active;
            session.active_series_uid = selected_series_uid.clone();
            session.active_frame_index = 0;
            session.selected_measurement_id = None;
            session.draft_measurement = None;
            session.handle_drag = None;
            session.quad_reference_drag = None;
            reset_viewport_state(&mut session, true);

            if !session.frames_by_series.contains_key(&selected_series_uid) {
                prepare_viewer_for_series_load(&session);
                if let Err(error) = update_series_model(&session)
                    .and_then(|_| update_measurements_model(&session))
                    .and_then(|_| update_measurement_overlays(&session))
                    .and_then(|_| apply_viewport_state(&session))
                {
                    info!("Failed to prepare viewer for series switch: {}", error);
                }
                true
            } else {
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
                false
            }
        };

        if should_load {
            start_async_series_load(&session_for_series, selected_series_uid);
        }
    });

    let session_for_tool = session.clone();
    viewer.on_tool_selected(move |tool| {
        let mut session = session_for_tool.borrow_mut();
        let had_draft = session.draft_measurement.is_some();
        session.active_tool = shared_viewer_tool(tool);
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
            viewer.set_active_tool(ui_viewer_tool(session.active_tool));
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
                        ViewerTool::WindowLevel | ViewerTool::Pan | ViewerTool::Zoom,
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
            ViewerTool::WindowLevel | ViewerTool::Pan | ViewerTool::Zoom => {
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
            ViewerTool::Line
            | ViewerTool::Angle
            | ViewerTool::RectangleRoi
            | ViewerTool::EllipseRoi => {
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
                    ViewerTool::Line => DraftMeasurement::Line {
                        start: point,
                        end: point,
                    },
                    ViewerTool::Angle => DraftMeasurement::Angle {
                        vertex: point,
                        arm1: point,
                        arm2: None,
                    },
                    ViewerTool::RectangleRoi => DraftMeasurement::Rectangle {
                        corner1: point,
                        corner2: point,
                    },
                    ViewerTool::EllipseRoi => DraftMeasurement::Ellipse {
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
            ViewerTool::Line | ViewerTool::RectangleRoi | ViewerTool::EllipseRoi => {
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
            ViewerTool::Angle => {
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
                ViewerTool::Pan => {
                    session.viewport_offset_x = drag_state.start_offset_x + dx;
                    session.viewport_offset_y = drag_state.start_offset_y + dy;
                    apply_viewport_state(&session)
                }
                ViewerTool::Zoom => {
                    let factor = (1.0 - dy * 0.01).max(0.1);
                    session.viewport_scale = (drag_state.start_scale * factor).clamp(0.25, 8.0);
                    apply_viewport_state(&session)
                }
                ViewerTool::WindowLevel => {
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
            ViewerTool::Pan => {
                session.viewport_offset_x = drag_state.start_offset_x + dx;
                session.viewport_offset_y = drag_state.start_offset_y + dy;
                apply_viewport_state(&session)
            }
            ViewerTool::Zoom => {
                let factor = (1.0 - dy * 0.01).max(0.1);
                session.viewport_scale = (drag_state.start_scale * factor).clamp(0.25, 8.0);
                apply_viewport_state(&session)
            }
            ViewerTool::WindowLevel => {
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
                    QuadReferenceTarget::AdjustSlab(line_kind) => {
                        session.quad_reference_drag = Some(QuadReferenceDrag::AdjustSlab {
                            view: kind,
                            line_kind,
                        });
                        let _ = apply_viewport_state(&session);
                        return;
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
                (ViewerTool::WindowLevel, Some((scalar_min, scalar_max))) => {
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
                QuadReferenceDrag::AdjustSlab { view, line_kind } => {
                    if view != kind {
                        return;
                    }
                    let slab_drag = session
                        .prepared_volumes_by_series
                        .get(&session.active_series_uid)
                        .and_then(|prepared| {
                            let bounds = prepared.world_bounds();
                            let pointer_world = quad_mpr_world_point_from_viewport(
                                &session,
                                kind,
                                x,
                                y,
                                viewport_width,
                                viewport_height,
                            )?;
                            let current_state =
                                quad_slice_view_state_for_kind(&session, prepared, kind)?;
                            let other_state =
                                quad_slice_view_state_for_kind(&session, prepared, line_kind)?;
                            let thickness = (pointer_world - current_state.crosshair_world(bounds))
                                .dot(other_state.slice_plane(bounds).normal())
                                .abs();
                            let slice_mode = line_kind.slice_mode()?;
                            let min_active_half_thickness =
                                (prepared.slice_scroll_step(slice_mode) * 0.5).max(0.25);
                            Some((slice_mode, thickness, min_active_half_thickness))
                        });
                    if let Some((slice_mode, thickness, min_active_half_thickness)) = slab_drag {
                        let slice_state = active_slice_view_state(&mut session);
                        if slice_state.mode != slice_mode {
                            slice_state.set_mode(slice_mode);
                        }
                        slice_state.set_slab_half_thickness_from_drag(
                            thickness,
                            min_active_half_thickness,
                            SliceProjectionMode::MaximumIntensity,
                        );
                        render_quad_mpr_previews(&mut session)
                    } else {
                        Ok(())
                    }
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
        if !matches!(session.active_tool, ViewerTool::WindowLevel) {
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
        let measurement_frame = active_measurement_frame(&session);
        let measurement_image = measurement_frame.as_ref().map(measurement_image_view);
        let value_text = measurement_overlay_text(
            &selected,
            active_pixel_spacing(&session),
            measurement_image.as_ref(),
        );

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
    let initial_series_uid = session.borrow().active_series_uid.clone();
    if !initial_series_uid.is_empty() {
        start_async_series_load(&session, initial_series_uid);
    }
    Ok(viewer)
}

fn start_async_series_load(session: &Rc<RefCell<ViewerSession>>, series_uid: String) {
    let imagebox = {
        let mut session_ref = session.borrow_mut();
        if let Some(timer) = session_ref.series_load_timer.take() {
            timer.stop();
        }
        session_ref.pending_series_load_uid = Some(series_uid.clone());
        prepare_viewer_for_series_load(&session_ref);
        session_ref.imagebox.as_ref().clone()
    };

    let requested_series_uid = series_uid.clone();
    let (sender, receiver) = mpsc::channel::<LeafResult<LoadedSeriesData>>();
    std::thread::spawn(move || {
        let _ = sender.send(load_series_data(&imagebox, &series_uid));
    });

    let session_for_timer = session.clone();
    let timer = Timer::default();
    timer.start(
        TimerMode::Repeated,
        Duration::from_millis(16),
        move || match receiver.try_recv() {
            Ok(result) => {
                let mut session_ref = session_for_timer.borrow_mut();
                if let Some(timer) = session_ref.series_load_timer.as_ref() {
                    timer.stop();
                }
                session_ref.series_load_timer = None;

                match result {
                    Ok(loaded) => {
                        let loaded_series_uid = loaded.series_uid.clone();
                        apply_loaded_series_data(&mut session_ref, loaded);

                        if session_ref.pending_series_load_uid.as_deref()
                            == Some(loaded_series_uid.as_str())
                        {
                            session_ref.pending_series_load_uid = None;
                        }

                        if session_ref.active_series_uid == loaded_series_uid {
                            if let Some(viewer) = session_ref.viewer.upgrade() {
                                set_viewer_loading_state(&viewer, false, "", "");
                            }
                            let result = if session_ref.volume_preview_active {
                                render_or_show_volume_preview(&mut session_ref)
                            } else {
                                update_viewer_image(&mut session_ref)
                                    .and_then(|_| update_measurements_model(&session_ref))
                                    .and_then(|_| update_measurement_overlays(&session_ref))
                            };
                            if let Err(error) = result {
                                if let Some(viewer) = session_ref.viewer.upgrade() {
                                    viewer.set_connection_status(
                                        format!("Failed to load series: {error}").into(),
                                    );
                                }
                                info!(
                                    "Failed to display loaded series {}: {}",
                                    loaded_series_uid, error
                                );
                            }
                        }
                    }
                    Err(error) => {
                        if session_ref.pending_series_load_uid.as_deref()
                            == Some(requested_series_uid.as_str())
                        {
                            session_ref.pending_series_load_uid = None;
                            if let Some(viewer) = session_ref.viewer.upgrade() {
                                set_viewer_loading_state(&viewer, false, "", "");
                                viewer.set_connection_status(
                                    format!("Failed to load series: {error}").into(),
                                );
                                viewer.set_slice_info("0/0".into());
                            }
                        }
                        info!("Failed to load series {}: {}", requested_series_uid, error);
                    }
                }
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                let mut session_ref = session_for_timer.borrow_mut();
                if let Some(timer) = session_ref.series_load_timer.as_ref() {
                    timer.stop();
                }
                session_ref.series_load_timer = None;
                if session_ref.pending_series_load_uid.as_deref()
                    == Some(requested_series_uid.as_str())
                {
                    session_ref.pending_series_load_uid = None;
                    if let Some(viewer) = session_ref.viewer.upgrade() {
                        set_viewer_loading_state(&viewer, false, "", "");
                        viewer.set_connection_status("Failed to load series".into());
                        viewer.set_slice_info("0/0".into());
                    }
                }
            }
        },
    );

    session.borrow_mut().series_load_timer = Some(timer);
}

fn load_series_data(imagebox: &Imagebox, series_uid: &str) -> LeafResult<LoadedSeriesData> {
    leaf_viewer::load_series_data(imagebox, series_uid)
}

fn apply_loaded_series_data(session: &mut ViewerSession, loaded: LoadedSeriesData) {
    leaf_viewer::apply_loaded_series_data(&mut session.state, loaded);
}

fn prepare_viewer_for_series_load(session: &ViewerSession) {
    let Some(viewer) = session.viewer.upgrade() else {
        return;
    };

    viewer.set_connection_status("Loading series...".into());
    viewer.set_viewport_image(slint::Image::default());
    viewer.set_slice_info("Loading...".into());
    viewer.set_measurements(empty_measurement_model());
    viewer.set_measurement_overlays(empty_measurement_overlay_model());
    viewer.set_orientation_top("".into());
    viewer.set_orientation_bottom("".into());
    viewer.set_orientation_left("".into());
    viewer.set_orientation_right("".into());
    set_viewer_loading_state(&viewer, true, "Loading series", "Loading series...");
}

fn set_viewer_loading_state(
    viewer: &leaf_ui::StudyViewerWindow,
    visible: bool,
    title: &str,
    message: &str,
) {
    viewer.set_loading_visible(visible);
    viewer.set_loading_title(title.into());
    viewer.set_loading_message(message.into());
    viewer.set_loading_current(0);
    viewer.set_loading_total(0);
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
                    .unwrap_or(series.num_instances as i32),
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
    let mut measurement_frames_by_slice = HashMap::new();
    let entries = session
        .measurements_by_series
        .get(&session.active_series_uid)
        .map(|measurements| {
            measurements
                .iter()
                .map(|measurement| {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        measurement_frames_by_slice.entry(measurement.slice_index)
                    {
                        if let Some(frame) =
                            measurement_frame_for_slice(session, measurement.slice_index)
                        {
                            entry.insert(frame);
                        }
                    }
                    let measurement_image = measurement_frames_by_slice
                        .get(&measurement.slice_index)
                        .map(measurement_image_view);
                    measurement_entry(measurement, pixel_spacing, measurement_image.as_ref())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    viewer.set_measurements(ModelRc::from(Rc::new(VecModel::from(entries))));
    Ok(())
}

fn active_volume_view_state(session: &mut ViewerSession) -> &mut VolumeViewState {
    let series_uid = session.active_series_uid.clone();
    session
        .volume_view_state_by_series
        .entry(series_uid)
        .or_default()
}

fn active_slice_view_state(session: &mut ViewerSession) -> &mut SlicePreviewState {
    let series_uid = session.active_series_uid.clone();
    session
        .slice_view_state_by_series
        .entry(series_uid)
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
    if let Some(slice_mode) = kind.slice_mode() {
        active_slice_view_state(session).set_mode(slice_mode);
    }
}

fn quad_preview(
    session: &ViewerSession,
    kind: QuadViewportKind,
) -> Option<&leaf_viewer::RgbaPreview> {
    leaf_viewer::quad_preview(session, kind)
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
    leaf_viewer::quad_world_to_viewport_point(plane, to_shared_viewport_geometry(geometry), world)
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

    leaf_viewer::build_quad_reference_lines(session)?;
    for kind in [
        QuadViewportKind::Axial,
        QuadViewportKind::Coronal,
        QuadViewportKind::Sagittal,
    ] {
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
    leaf_viewer::normalized_angle_delta(current_angle_rad, start_angle_rad)
}

fn point_to_segment_distance_sq(
    point_x: f32,
    point_y: f32,
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
) -> f32 {
    leaf_viewer::point_to_segment_distance_sq(point_x, point_y, start_x, start_y, end_x, end_y)
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

    if let Some((_, source_kind)) = session
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
        } | QuadReferenceDrag::AdjustSlab {
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
                | QuadReferenceTarget::RotateLine(line_kind)
                | QuadReferenceTarget::AdjustSlab(line_kind),
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
    leaf_viewer::next_volume_blend_mode(mode)
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
            h3_x: overlay.handle3_x,
            h3_y: overlay.handle3_y,
            h4_x: overlay.handle4_x,
            h4_y: overlay.handle4_y,
            kind: overlay.source_kind.index(),
            active: quad_reference_line_highlighted(session, kind, overlay.source_kind),
            slab_active: overlay.slab_active,
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
    measurement_image: Option<&MeasurementImage<'_>>,
) -> leaf_ui::MeasurementEntry {
    leaf_ui::MeasurementEntry {
        id: measurement.id.clone().into(),
        kind: measurement_kind_label(measurement).into(),
        value: measurement_panel_value_text(measurement, pixel_spacing, measurement_image).into(),
        label: measurement.label.clone().unwrap_or_default().into(),
        slice_index: (measurement.slice_index + 1) as i32,
    }
}

fn measurement_kind_label(measurement: &Measurement) -> &'static str {
    leaf_viewer::measurement_kind_label(measurement)
}

fn measurement_overlay_text(
    measurement: &Measurement,
    pixel_spacing: (f64, f64),
    measurement_image: Option<&MeasurementImage<'_>>,
) -> String {
    leaf_viewer::measurement_overlay_text(measurement, pixel_spacing, measurement_image)
}

fn measurement_panel_value_text(
    measurement: &Measurement,
    pixel_spacing: (f64, f64),
    measurement_image: Option<&MeasurementImage<'_>>,
) -> String {
    leaf_viewer::measurement_panel_value_text(measurement, pixel_spacing, measurement_image)
}

fn active_pixel_spacing(session: &ViewerSession) -> (f64, f64) {
    session
        .series
        .iter()
        .find(|series| series.series_uid.0 == session.active_series_uid)
        .and_then(|series| series.pixel_spacing)
        .unwrap_or((1.0, 1.0))
}

fn measurement_frame_for_slice(
    session: &ViewerSession,
    slice_index: usize,
) -> Option<MeasurementFrame> {
    let frame_ref = session
        .frames_by_series
        .get(&session.active_series_uid)?
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

fn active_measurement_frame(session: &ViewerSession) -> Option<MeasurementFrame> {
    measurement_frame_for_slice(session, session.active_frame_index)
}

fn measurement_image_view<'a>(frame: &'a MeasurementFrame) -> MeasurementImage<'a> {
    MeasurementImage {
        width: frame.width,
        height: frame.height,
        pixels: &frame.pixels,
        unit: &frame.unit,
    }
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
    let measurement_frame = active_measurement_frame(session);
    let measurement_image = measurement_frame.as_ref().map(measurement_image_view);

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
                measurement_image.as_ref(),
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
        if let Some(overlay) = measurement_to_overlay(
            session,
            &temp,
            pixel_spacing,
            measurement_image.as_ref(),
            false,
            true,
        ) {
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
    measurement_image: Option<&MeasurementImage<'_>>,
    selected: bool,
    draft: bool,
) -> Option<leaf_ui::MeasurementOverlay> {
    let label = measurement_overlay_text(measurement, pixel_spacing, measurement_image);
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
    let measurement_frame = active_measurement_frame(session);
    let measurement_image = measurement_frame.as_ref().map(measurement_image_view);
    let value_text = measurement_overlay_text(
        &measurement,
        active_pixel_spacing(session),
        measurement_image.as_ref(),
    );
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

fn apply_overlays_to_rgba(
    rgba: &mut [u8],
    image_width: u32,
    image_height: u32,
    overlays: &[OverlayBitmap],
) {
    leaf_viewer::apply_overlays_to_rgba(rgba, image_width, image_height, overlays)
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
    leaf_viewer::transformed_image_dimensions(width, height, to_shared_transform(transform))
}

fn source_to_display_point_raw(
    point: DVec2,
    source_width: f64,
    source_height: f64,
    transform: ImageTransformState,
) -> DVec2 {
    leaf_viewer::source_to_display_point_raw(
        point,
        source_width,
        source_height,
        to_shared_transform(transform),
    )
}

fn display_to_source_point_raw(
    point: DVec2,
    source_width: f64,
    source_height: f64,
    transform: ImageTransformState,
) -> DVec2 {
    leaf_viewer::display_to_source_point_raw(
        point,
        source_width,
        source_height,
        to_shared_transform(transform),
    )
}

fn transform_rgba(
    rgba: &[u8],
    source_width: u32,
    source_height: u32,
    transform: ImageTransformState,
) -> Vec<u8> {
    leaf_viewer::transform_rgba(
        rgba,
        source_width,
        source_height,
        to_shared_transform(transform),
    )
}

fn orientation_labels_for_frame(
    orientation: Option<[f64; 6]>,
    transform: ImageTransformState,
) -> (String, String, String, String) {
    leaf_viewer::orientation_labels_for_frame(orientation, to_shared_transform(transform))
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
    leaf_viewer::orientation_labels_for_slice_plane(plane)
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
    leaf_viewer::mpr_uv_from_viewport(to_shared_viewport_geometry(geometry), x, y)
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
    leaf_viewer::displayed_image_geometry(
        viewport_width,
        viewport_height,
        frame_width,
        frame_height,
        viewport_scale,
        viewport_offset_x,
        viewport_offset_y,
    )
    .map(from_shared_viewport_geometry)
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

fn apply_viewport_state(session: &ViewerSession) -> LeafResult<()> {
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;

    viewer.set_active_tool(ui_viewer_tool(session.active_tool));
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
    let preview = leaf_viewer::render_volume_image(session, interactive)?;
    show_volume_preview(session, &preview)
}

fn render_slice_preview(session: &mut ViewerSession) -> LeafResult<()> {
    let preview = leaf_viewer::render_slice_image(session)?;
    show_volume_preview(session, &preview)
}

fn render_quad_preview_for_kind(
    session: &mut ViewerSession,
    kind: QuadViewportKind,
    interactive: bool,
) -> LeafResult<AdvancedViewportPreview> {
    let preview = leaf_viewer::render_quad_rgba(session, kind, interactive)?;
    session.quad_previews_by_kind.insert(kind, preview.clone());
    let image = leaf_ui::image_from_rgba8(preview.width, preview.height, preview.rgba)
        .map_err(|error| LeafError::Render(error.to_string()))?;
    Ok(AdvancedViewportPreview {
        image,
        info: preview.info,
    })
}

fn apply_quad_preview_to_viewer(
    viewer: &leaf_ui::StudyViewerWindow,
    kind: QuadViewportKind,
    preview: &AdvancedViewportPreview,
) -> LeafResult<()> {
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
    Ok(())
}

fn render_quad_single_preview(
    session: &mut ViewerSession,
    kind: QuadViewportKind,
    interactive: bool,
) -> LeafResult<()> {
    let preview = render_quad_preview_for_kind(session, kind, interactive)?;
    session.volume_preview_active = true;
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    apply_quad_preview_to_viewer(&viewer, kind, &preview)?;
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
        apply_quad_preview_to_viewer(&viewer, kind, &preview)?;
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
        apply_quad_preview_to_viewer(&viewer, kind, &preview)?;
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
    leaf_viewer::reset_viewport_state(session, clear_defaults);
}

fn to_rgba(frame: &leaf_dicom::pixel::DecodedFrame, lut_name: &str) -> LeafResult<Vec<u8>> {
    leaf_viewer::to_rgba(frame, lut_name)
}
