//! Local viewer state and rendering flow for pacsleaf.

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

struct ViewerSession {
    viewer: slint::Weak<leaf_ui::StudyViewerWindow>,
    series: Vec<SeriesInfo>,
    instances_by_series: std::collections::HashMap<String, Vec<InstanceInfo>>,
    frames_by_series: HashMap<String, Vec<FrameRef>>,
    measurements_by_series: HashMap<String, Vec<Measurement>>,
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
    start_view_state: VolumeViewState,
}

#[derive(Clone)]
struct DraftMeasurement {
    start: DVec2,
    end: DVec2,
}

#[derive(Clone, Copy)]
struct ViewportGeometry {
    image_origin_x: f32,
    image_origin_y: f32,
    image_width: f32,
    image_height: f32,
}

pub(crate) fn open_viewer_for_study(
    imagebox: &Imagebox,
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
    let measurements_by_series = HashMap::new();

    let viewer =
        leaf_ui::StudyViewerWindow::new().map_err(|error| LeafError::Render(error.to_string()))?;
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
        series,
        instances_by_series,
        frames_by_series,
        measurements_by_series,
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
    }

    let session_for_series = session.clone();
    viewer.on_series_selected(move |series_uid| {
        let mut session = session_for_series.borrow_mut();
        let preview_active = session.volume_preview_active;
        session.active_series_uid = series_uid.to_string();
        session.active_frame_index = 0;
        session.selected_measurement_id = None;
        session.draft_measurement = None;
        reset_viewport_state(&mut session, true);
        let result = update_series_model(&session).and_then(|_| {
            if preview_active {
                render_or_show_volume_preview(&mut session)
            } else {
                update_viewer_image(&mut session).and_then(|_| update_measurements_model(&session))
            }
        });
        if let Err(error) = result {
            info!("Failed to switch series: {}", error);
        }
    });

    let session_for_tool = session.clone();
    viewer.on_tool_selected(move |tool| {
        let mut session = session_for_tool.borrow_mut();
        session.active_tool = tool;
        session.drag_state = None;
        session.volume_drag_state = None;
        session.draft_measurement = None;
        if let Err(error) = apply_viewport_state(&session) {
            info!("Failed to switch tool: {}", error);
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
    viewer.on_viewport_mouse_down(move |x, y, viewport_width, viewport_height| {
        let mut session = session_for_mouse_down.borrow_mut();
        update_viewport_dimensions(&mut session, viewport_width, viewport_height);
        if session.volume_preview_active {
            session.drag_state = None;
            session.volume_drag_state = Some(VolumeDragState {
                origin_x: x,
                origin_y: y,
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
            leaf_ui::ViewerTool::Line => {
                session.drag_state = None;
                session.draft_measurement = viewport_to_image_point(&session, x, y, false)
                    .map(|start| DraftMeasurement { start, end: start });
                if let Err(error) = update_measurement_overlays(&session) {
                    info!("Failed to begin line measurement: {}", error);
                }
            }
            _ => {
                session.drag_state = None;
                session.draft_measurement = None;
            }
        }
    });

    let session_for_mouse_up = session.clone();
    viewer.on_viewport_mouse_up(move |x, y, viewport_width, viewport_height| {
        let mut session = session_for_mouse_up.borrow_mut();
        update_viewport_dimensions(&mut session, viewport_width, viewport_height);
        if session.volume_preview_active {
            session.volume_drag_state = None;
            return;
        }

        if matches!(session.active_tool, leaf_ui::ViewerTool::Line) {
            if let Some(end) = viewport_to_image_point(&session, x, y, true) {
                if let Some(draft) = session.draft_measurement.as_mut() {
                    draft.end = end;
                }
            }

            if let Err(error) = finalize_line_measurement(&mut session) {
                info!("Failed to finalize line measurement: {}", error);
            }
        } else {
            session.drag_state = None;
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
            apply_volume_drag(&mut session, drag_state.start_view_state, dx, dy);
            if let Err(error) = render_or_show_volume_preview(&mut session) {
                info!("Failed to update volume preview interaction: {}", error);
            }
            return;
        }

        if matches!(session.active_tool, leaf_ui::ViewerTool::Line) {
            if let Some(end) = viewport_to_image_point(&session, x, y, true) {
                if let Some(draft) = session.draft_measurement.as_mut() {
                    draft.end = end;
                }
            }

            if let Err(error) = update_measurement_overlays(&session) {
                info!("Failed to update line measurement preview: {}", error);
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

        let result =
            update_viewer_image(&mut session).and_then(|_| update_measurements_model(&session));
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

    let session_for_volume_toggle = session.clone();
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

    viewer
        .show()
        .map_err(|error| LeafError::Render(error.to_string()))?;
    Ok(viewer)
}

fn update_series_model(session: &ViewerSession) -> LeafResult<()> {
    let entries = session
        .series
        .iter()
        .map(|series| leaf_ui::SeriesThumbnail {
            series_uid: series.series_uid.0.clone().into(),
            series_number: series.series_number.unwrap_or_default(),
            modality: series.modality.clone().into(),
            description: series
                .series_description
                .clone()
                .unwrap_or_else(|| "-".to_string())
                .into(),
            instance_count: session
                .instances_by_series
                .get(&series.series_uid.0)
                .map(|instances| instances.len() as i32)
                .unwrap_or(0),
            active: series.series_uid.0 == session.active_series_uid,
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
) {
    let active_tool = session.active_tool;
    let view_state = active_volume_view_state(session);
    *view_state = start_view_state;
    match active_tool {
        leaf_ui::ViewerTool::Pan => view_state.pan(delta_x as f64, delta_y as f64),
        leaf_ui::ViewerTool::Zoom => {
            let factor = (1.0 - delta_y as f64 * 0.01).clamp(0.25, 4.0);
            view_state.zoom_by(factor);
        }
        _ => view_state.orbit(delta_x as f64 * 0.45, delta_y as f64 * 0.35),
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
    let pixel_spacing = active_pixel_spacing(session);

    let mut overlays = session
        .measurements_by_series
        .get(&session.active_series_uid)
        .into_iter()
        .flat_map(|measurements| measurements.iter())
        .filter(|measurement| measurement.slice_index == session.active_frame_index)
        .filter_map(|measurement| {
            let label = measurement_value_text(measurement, pixel_spacing);
            measurement_overlay(
                session,
                &measurement.id,
                measurement_line_points(measurement)?,
                label,
                session.selected_measurement_id.as_deref() == Some(measurement.id.as_str()),
                false,
            )
        })
        .collect::<Vec<_>>();

    if let Some(draft) = session.draft_measurement.as_ref() {
        let draft_distance = line_distance_mm(draft.start, draft.end, pixel_spacing);
        if let Some(overlay) = measurement_overlay(
            session,
            "draft-line",
            (draft.start, draft.end),
            format_distance_mm(draft_distance),
            false,
            true,
        ) {
            overlays.push(overlay);
        }
    }

    viewer.set_measurement_overlays(ModelRc::from(Rc::new(VecModel::from(overlays))));
    Ok(())
}

fn measurement_line_points(measurement: &Measurement) -> Option<(DVec2, DVec2)> {
    match &measurement.kind {
        MeasurementKind::Line { start, end } => Some((*start, *end)),
        _ => None,
    }
}

fn measurement_overlay(
    session: &ViewerSession,
    id: &str,
    points: (DVec2, DVec2),
    label: String,
    selected: bool,
    draft: bool,
) -> Option<leaf_ui::MeasurementOverlay> {
    let (start, end) = points;
    let (start_x, start_y) = image_to_viewport_point(session, start)?;
    let (end_x, end_y) = image_to_viewport_point(session, end)?;
    let dx = end_x - start_x;
    let dy = end_y - start_y;
    if (dx * dx + dy * dy).sqrt() < 1.0 {
        return None;
    }

    Some(leaf_ui::MeasurementOverlay {
        id: id.into(),
        commands: format!("M {start_x:.2} {start_y:.2} L {end_x:.2} {end_y:.2}").into(),
        label_x: end_x + 6.0,
        label_y: (end_y - 16.0).max(6.0),
        label: label.into(),
        selected,
        draft,
    })
}

fn finalize_line_measurement(session: &mut ViewerSession) -> LeafResult<()> {
    session.drag_state = None;

    let Some(draft) = session.draft_measurement.take() else {
        return update_measurement_overlays(session);
    };

    if (draft.end - draft.start).length() < 2.0 {
        return update_measurement_overlays(session);
    }

    let measurement = Measurement::line(
        &session.active_series_uid,
        session.active_frame_index,
        draft.start,
        draft.end,
    );
    let value_text = measurement_value_text(&measurement, active_pixel_spacing(session));
    let active_series_uid = session.active_series_uid.clone();
    session.selected_measurement_id = Some(measurement.id.clone());
    session
        .measurements_by_series
        .entry(active_series_uid)
        .or_default()
        .push(measurement.clone());

    if !session.measurement_panel_visible {
        session.measurement_panel_visible = true;
        if let Some(viewer) = session.viewer.upgrade() {
            viewer.set_measurement_panel_visible(true);
        }
    }

    update_measurements_model(session)?;
    update_measurement_overlays(session)?;

    if let Some(viewer) = session.viewer.upgrade() {
        viewer.set_connection_status(format!("Created line {}", value_text).into());
    }

    Ok(())
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

fn line_distance_mm(start: DVec2, end: DVec2, pixel_spacing: (f64, f64)) -> f64 {
    let dx = (end.x - start.x) * pixel_spacing.1;
    let dy = (end.y - start.y) * pixel_spacing.0;
    (dx * dx + dy * dy).sqrt()
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
        let volume_zoom = session
            .volume_view_state_by_series
            .get(&session.active_series_uid)
            .map(|state| state.zoom)
            .unwrap_or(1.0);
        viewer.set_viewport_scale(1.0);
        viewer.set_viewport_offset_x(0.0);
        viewer.set_viewport_offset_y(0.0);
        viewer.set_window_info("DVR orbit".into());
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
        renderer.render_prepared_preview(prepared, &view_state, preview_size.0, preview_size.1)?
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

    let max_side = width.max(height);
    if max_side > 768 {
        let scale = 768.0 / max_side as f32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use leaf_core::domain::{SeriesUid, SopInstanceUid, StudyUid};
    use leaf_tools::measurement::Measurement;

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
    fn formats_line_measurement_values_in_millimeters() {
        let measurement =
            Measurement::line("series", 0, DVec2::new(0.0, 0.0), DVec2::new(3.0, 4.0));

        assert_eq!(measurement_value_text(&measurement, (1.0, 1.0)), "5.00 mm");
    }
}
