//! Local viewer state and rendering flow for pacsleaf.

use crate::browser::{
    apply_window_geometry, capture_window_geometry, load_window_geometry, save_window_geometry,
    VIEWER_WINDOW_GEOMETRY_KEY,
};
use glam::DVec2;
use leaf_core::domain::{InstanceInfo, SeriesInfo, SeriesUid, StudyUid};
use leaf_core::error::{LeafError, LeafResult};
use leaf_db::imagebox::Imagebox;
use leaf_dicom::metadata::read_instance_geometry;
use leaf_dicom::pixel::{decode_frame_with_window, frame_count};
use leaf_render::{PreparedVolume, VolumePreviewImage, VolumePreviewRenderer, VolumeViewState};
use leaf_tools::measurement::{Measurement, MeasurementKind, MeasurementValue};
use slint::{ComponentHandle, ModelRc, VecModel};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::rc::Rc;
use tracing::info;

const THUMB_SIZE: usize = 64;

struct ViewerSession {
    viewer: slint::Weak<leaf_ui::StudyViewerWindow>,
    imagebox: Rc<Imagebox>,
    series: Vec<SeriesInfo>,
    instances_by_series: std::collections::HashMap<String, Vec<InstanceInfo>>,
    frames_by_series: HashMap<String, Vec<FrameRef>>,
    measurements_by_series: HashMap<String, Vec<Measurement>>,
    thumbnails_by_series: HashMap<String, slint::Image>,
    active_series_uid: String,
    active_frame_index: usize,
    measurement_panel_visible: bool,
    volume_preview_active: bool,
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
    selected_measurement_id: Option<String>,
    draft_measurement: Option<DraftMeasurement>,
    handle_drag: Option<HandleDrag>,
    drag_state: Option<ViewportDragState>,
    volume_drag_state: Option<VolumeDragState>,
    volume_renderer: Option<VolumePreviewRenderer>,
    prepared_volumes_by_series: HashMap<String, PreparedVolume>,
    volume_view_state_by_series: HashMap<String, VolumeViewState>,
}

#[derive(Clone)]
struct FrameRef {
    file_path: String,
    frame_index: u32,
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
        active_series_uid,
        active_frame_index: 0,
        measurement_panel_visible: false,
        volume_preview_active: false,
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
        selected_measurement_id: None,
        draft_measurement: None,
        handle_drag: None,
        drag_state: None,
        volume_drag_state: None,
        volume_renderer: None,
        prepared_volumes_by_series: HashMap::new(),
        volume_view_state_by_series: HashMap::new(),
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

    let session_for_scroll = session.clone();
    viewer.on_viewport_scroll(move |delta| {
        let mut session = session_for_scroll.borrow_mut();
        if session.volume_preview_active {
            let zoom_factor = if delta > 0.0 {
                1.1
            } else if delta < 0.0 {
                0.9
            } else {
                1.0
            };
            active_volume_view_state(&mut session).zoom_by(zoom_factor);
            if let Err(error) = render_or_show_volume_preview(&mut session) {
                info!("Failed to zoom volume preview: {}", error);
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
            session.drag_state = None;
            session.volume_drag_state = Some(VolumeDragState {
                origin_x: x,
                origin_y: y,
                button,
                start_view_state: *active_volume_view_state(&mut session),
            });
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
            session.volume_drag_state = None;
            if let Err(error) = render_volume_preview(&mut session, false) {
                info!("Failed to finalize volume preview interaction: {}", error);
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
        if session.volume_preview_active {
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
            active_volume_view_state(&mut session).reset();
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

    let session_for_volume_toggle = session.clone();
    let viewer_weak_for_close = viewer.as_weak();
    let imagebox_for_close = imagebox.clone();
    viewer.on_toggle_volume_preview(move || {
        let mut session = session_for_volume_toggle.borrow_mut();
        let result = if session.volume_preview_active {
            session.volume_preview_active = false;
            session.volume_drag_state = None;
            update_viewer_image(&mut session).and_then(|_| update_measurements_model(&session))
        } else {
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

fn image_to_viewport_point(session: &ViewerSession, point: DVec2) -> Option<(f32, f32)> {
    let geometry = current_viewport_geometry(session)?;
    let frame_width = session.active_frame_width as f32;
    let frame_height = session.active_frame_height as f32;
    if frame_width <= 0.0 || frame_height <= 0.0 {
        return None;
    }

    Some((
        geometry.image_origin_x + (point.x as f32 / frame_width) * geometry.image_width,
        geometry.image_origin_y + (point.y as f32 / frame_height) * geometry.image_height,
    ))
}

fn viewport_to_image_point(session: &ViewerSession, x: f32, y: f32, clamp: bool) -> Option<DVec2> {
    let geometry = current_viewport_geometry(session)?;
    let frame_width = session.active_frame_width as f32;
    let frame_height = session.active_frame_height as f32;
    if frame_width <= 0.0 || frame_height <= 0.0 {
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

    Some(DVec2::new(
        normalized_x as f64 * frame_width as f64,
        normalized_y as f64 * frame_height as f64,
    ))
}

fn current_viewport_geometry(session: &ViewerSession) -> Option<ViewportGeometry> {
    displayed_image_geometry(
        session.viewport_width,
        session.viewport_height,
        session.active_frame_width,
        session.active_frame_height,
        session.viewport_scale,
        session.viewport_offset_x,
        session.viewport_offset_y,
    )
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
        .ok_or_else(|| LeafError::NoData("Frame index out of range".into()))?;
    let frame = decode_frame_with_window(
        Path::new(&frame_ref.file_path),
        frame_ref.frame_index,
        session.window_center.zip(session.window_width),
    )?;

    if session.default_window_center.is_none() || session.default_window_width.is_none() {
        session.default_window_center = Some(frame.window_center);
        session.default_window_width = Some(frame.window_width);
    }
    if session.window_center.is_none() || session.window_width.is_none() {
        session.window_center = Some(frame.window_center);
        session.window_width = Some(frame.window_width);
    }
    session.active_frame_width = frame.width;
    session.active_frame_height = frame.height;
    let rgba = to_rgba(&frame)?;

    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    viewer.set_viewport_image(
        leaf_ui::image_from_rgba8(frame.width, frame.height, rgba)
            .map_err(|error| LeafError::Render(error.to_string()))?,
    );
    viewer.set_slice_info(format!("{}/{}", session.active_frame_index + 1, total).into());
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

    if session.volume_preview_active {
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
    render_volume_preview(session, false)
}

fn render_volume_preview(session: &mut ViewerSession, interactive: bool) -> LeafResult<()> {
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

fn preview_dimensions(session: &ViewerSession) -> (u32, u32) {
    let mut width = if session.viewport_width > 0.0 {
        session.viewport_width.round() as u32
    } else {
        768
    };
    let mut height = if session.viewport_height > 0.0 {
        session.viewport_height.round() as u32
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

fn show_volume_preview(
    session: &mut ViewerSession,
    preview: &VolumePreviewImage,
) -> LeafResult<()> {
    session.volume_preview_active = true;
    session.active_frame_width = preview.width;
    session.active_frame_height = preview.height;
    let viewer = session
        .viewer
        .upgrade()
        .ok_or_else(|| LeafError::Render("Viewer window no longer available".into()))?;
    viewer.set_viewport_image(
        leaf_ui::image_from_rgba8(preview.width, preview.height, preview.rgba.clone())
            .map_err(|error| LeafError::Render(error.to_string()))?,
    );
    viewer.set_connection_status(
        "Local imagebox | DVR preview (drag=orbit, Pan/Zoom tools=3D)".into(),
    );
    viewer.set_window_info("DVR preview".into());
    viewer.set_slice_info("3D".into());
    apply_viewport_state(session)?;
    update_measurements_model(session)?;
    Ok(())
}

fn reset_viewport_state(session: &mut ViewerSession, clear_defaults: bool) {
    session.viewport_scale = 1.0;
    session.viewport_offset_x = 0.0;
    session.viewport_offset_y = 0.0;
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

fn to_rgba(frame: &leaf_dicom::pixel::DecodedFrame) -> LeafResult<Vec<u8>> {
    let expected = frame.width as usize * frame.height as usize;
    match frame.channels {
        1 => {
            if frame.pixels.len() != expected {
                return Err(LeafError::Render("Unexpected grayscale frame size".into()));
            }
            let mut rgba = Vec::with_capacity(expected * 4);
            for value in &frame.pixels {
                rgba.extend_from_slice(&[*value, *value, *value, 255]);
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
    fn preview_dimensions_keep_output_size_stable() {
        let session = ViewerSession {
            viewer: slint::Weak::default(),
            imagebox: test_imagebox(),
            series: Vec::new(),
            instances_by_series: HashMap::new(),
            frames_by_series: HashMap::new(),
            measurements_by_series: HashMap::new(),
            thumbnails_by_series: HashMap::new(),
            active_series_uid: String::new(),
            active_frame_index: 0,
            measurement_panel_visible: false,
            volume_preview_active: false,
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
            selected_measurement_id: None,
            draft_measurement: None,
            handle_drag: None,
            drag_state: None,
            volume_drag_state: None,
            volume_renderer: None,
            prepared_volumes_by_series: HashMap::new(),
            volume_view_state_by_series: HashMap::new(),
        };

        let preview = preview_dimensions(&session);

        assert_eq!(preview.0.max(preview.1), 640);
    }

    #[test]
    fn formats_line_measurement_values_in_millimeters() {
        let measurement =
            Measurement::line("series", 0, DVec2::new(0.0, 0.0), DVec2::new(3.0, 4.0));

        assert_eq!(measurement_value_text(&measurement, (1.0, 1.0)), "5.00 mm");
    }
}
