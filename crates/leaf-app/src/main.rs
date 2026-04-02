//! pacsleaf — Pure Rust Medical Imaging Viewer
//!
//! Companion viewer for pacsnode. Provides a fast, modern, dark-themed
//! clinical workstation for radiologists.

mod browser;
mod viewer;

use anyhow::Result;
use browser::{
    apply_browser_settings, apply_window_geometry, capture_window_geometry,
    default_dimse_port, default_local_ae_title, find_node, load_browser_settings,
    load_window_geometry, parse_remote_study_ref, qido_first_int, qido_first_string,
    refresh_browser, save_browser_settings, save_window_geometry, validate_browser_settings,
    BrowserQuery, BrowserSettings, RemoteStudyRef, BROWSER_WINDOW_GEOMETRY_KEY,
    VIEWER_WINDOW_GEOMETRY_KEY,
};
use image::load_from_memory;
use leaf_core::config::{data_dir, PacsNodeConfig};
use leaf_core::domain::{InstanceInfo, SeriesInfo, StudyInfo};
use leaf_core::error::{LeafError, LeafResult};
use leaf_db::imagebox::Imagebox;
use leaf_dicom::metadata::import_dicom_file;
use leaf_dicom::pixel::decode_frame;
use leaf_net::dicomweb::DicomWebClient;
use rfd::FileDialog;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use tracing::info;
use tracing_subscriber::EnvFilter;
use viewer::{install_viewer_tool_state, open_viewer_for_study};

struct RemoteSeriesRef {
    series_uid: String,
    series_number: i32,
    modality: String,
    description: String,
    instance_count: i32,
}

fn main() -> Result<()> {
    // Initialize logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("pacsleaf v{} starting", env!("CARGO_PKG_VERSION"));

    // Open the local imagebox database.
    let db_path = data_dir().join("imagebox.redb");
    let imagebox = Rc::new(Imagebox::open(&db_path)?);
    let settings_state = Rc::new(RefCell::new(load_browser_settings(&imagebox)?));
    let runtime = Rc::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?,
    );

    for import_path in cli_import_paths() {
        let summary = import_from_path(&imagebox, &import_path)?;
        info!(
            "Imported {} studies from {}",
            summary.study_count,
            import_path.display()
        );
    }

    // Create the event bus.
    let _event_bus = leaf_core::event::EventBus::default();

    // Launch the Slint UI.
    let browser = leaf_ui::StudyBrowserWindow::new()?;
    if let Some(geometry) = load_window_geometry(&imagebox, BROWSER_WINDOW_GEOMETRY_KEY)? {
        apply_window_geometry(browser.window(), geometry);
    }
    apply_browser_settings(&browser, &settings_state.borrow());
    browser.set_connection_status("Local imagebox".into());
    refresh_browser(
        &browser,
        &imagebox,
        &settings_state.borrow().pacs_nodes(),
        browser.get_network_mode(),
        &runtime,
        BrowserQuery::default(),
    )?;

    let open_windows: Rc<RefCell<Vec<leaf_ui::StudyViewerWindow>>> =
        Rc::new(RefCell::new(Vec::new()));

    let imagebox_for_browser_close = imagebox.clone();
    let browser_weak_for_close = browser.as_weak();
    browser.window().on_close_requested(move || {
        if let Some(browser) = browser_weak_for_close.upgrade() {
            if let Some(geometry) = capture_window_geometry(browser.window()) {
                if let Err(error) = save_window_geometry(
                    &imagebox_for_browser_close,
                    BROWSER_WINDOW_GEOMETRY_KEY,
                    &geometry,
                ) {
                    info!("Failed to save browser window geometry: {}", error);
                }
            }
        }
        slint::CloseRequestResponse::HideWindow
    });

    // Wire up callbacks.
    let browser_weak = browser.as_weak();
    let imagebox_for_search = imagebox.clone();
    let settings_for_search = settings_state.clone();
    let runtime_for_search = runtime.clone();
    browser.on_search(move |name, id, accession| {
        let browser = browser_weak.upgrade().unwrap();
        info!("Search: name={}, id={}, accession={}", name, id, accession);
        if let Err(error) = refresh_browser(
            &browser,
            &imagebox_for_search,
            &settings_for_search.borrow().pacs_nodes(),
            browser.get_network_mode(),
            &runtime_for_search,
            BrowserQuery::new(name.as_str(), id.as_str(), accession.as_str()),
        ) {
            browser.set_connection_status(format!("Search failed: {error}").into());
        }
    });

    let imagebox_for_open = imagebox.clone();
    let windows_for_open = open_windows.clone();
    let browser_for_open = browser.as_weak();
    let settings_for_open = settings_state.clone();
    let runtime_for_open = runtime.clone();
    browser.on_open_study(move |study_uid| {
        info!("Opening study: {}", study_uid);
        let result = if let Some(remote) = parse_remote_study_ref(study_uid.as_str()) {
            open_remote_viewer_for_study(
                &imagebox_for_open,
                &settings_for_open.borrow().pacs_nodes(),
                &runtime_for_open,
                &remote,
            )
        } else {
            open_viewer_for_study(&imagebox_for_open, study_uid.as_str())
        };
        match result {
            Ok(viewer) => windows_for_open.borrow_mut().push(viewer),
            Err(error) => {
                if let Some(browser) = browser_for_open.upgrade() {
                    browser.set_connection_status(format!("Open failed: {error}").into());
                }
                info!("Failed to open study {}: {}", study_uid, error);
            }
        }
    });

    let browser_for_import = browser.as_weak();
    let imagebox_for_import = imagebox.clone();
    let settings_for_import = settings_state.clone();
    let runtime_for_import = runtime.clone();
    browser.on_import_files(move || {
        let initial_directory = settings_for_import.borrow().initial_import_directory();
        let mut dialog = FileDialog::new();
        if initial_directory.exists() {
            dialog = dialog.set_directory(&initial_directory);
        }

        let Some(import_path) = dialog.pick_folder() else {
            if let Some(browser) = browser_for_import.upgrade() {
                browser.set_connection_status("Import cancelled".into());
            }
            return;
        };

        info!("Import files requested from {}", import_path.display());

        let result = (|| -> LeafResult<()> {
            let mut new_settings = settings_for_import.borrow().clone();
            new_settings.last_import_path = Some(import_path.display().to_string());
            save_browser_settings(&imagebox_for_import, &new_settings)?;
            *settings_for_import.borrow_mut() = new_settings.clone();

            import_from_path(&imagebox_for_import, import_path.as_ref()).and_then(|summary| {
                if let Some(browser) = browser_for_import.upgrade() {
                    let nodes = new_settings.pacs_nodes();
                    refresh_browser(
                        &browser,
                        &imagebox_for_import,
                        &nodes,
                        browser.get_network_mode(),
                        &runtime_for_import,
                        BrowserQuery::default(),
                    )?;
                    let status = if summary.file_count == 0 {
                        format!("No DICOM files found in {}", import_path.display())
                    } else {
                        format!(
                            "Imported {} files from {} into {} studies",
                            summary.file_count,
                            import_path.display(),
                            summary.study_count
                        )
                    };
                    browser.set_connection_status(status.into());
                }
                Ok(())
            })
        })();
        if let Err(error) = result {
            if let Some(browser) = browser_for_import.upgrade() {
                browser.set_connection_status(format!("Import failed: {error}").into());
            }
            info!("Import failed: {}", error);
        }
    });

    let browser_for_local = browser.as_weak();
    let imagebox_for_local = imagebox.clone();
    let settings_for_local = settings_state.clone();
    let runtime_for_local = runtime.clone();
    browser.on_activate_local(move || {
        if let Some(browser) = browser_for_local.upgrade() {
            browser.set_network_mode(false);
            let _ = refresh_browser(
                &browser,
                &imagebox_for_local,
                &settings_for_local.borrow().pacs_nodes(),
                false,
                &runtime_for_local,
                BrowserQuery::default(),
            );
            browser.set_connection_status("Local imagebox".into());
        }
    });

    let browser_for_network = browser.as_weak();
    let imagebox_for_network = imagebox.clone();
    let settings_for_network = settings_state.clone();
    let runtime_for_network = runtime.clone();
    browser.on_activate_network(move || {
        if let Some(browser) = browser_for_network.upgrade() {
            browser.set_network_mode(true);
            let result = refresh_browser(
                &browser,
                &imagebox_for_network,
                &settings_for_network.borrow().pacs_nodes(),
                true,
                &runtime_for_network,
                BrowserQuery::default(),
            );
            if let Err(error) = result {
                browser.set_connection_status(format!("Network enable failed: {error}").into());
            }
        }
    });

    let browser_for_toggle_settings = browser.as_weak();
    browser.on_toggle_settings(move || {
        if let Some(browser) = browser_for_toggle_settings.upgrade() {
            browser.set_settings_visible(!browser.get_settings_visible());
        }
    });

    let browser_for_save_settings = browser.as_weak();
    let imagebox_for_save_settings = imagebox.clone();
    let settings_for_save = settings_state.clone();
    let runtime_for_save = runtime.clone();
    browser.on_save_settings(
        move |node_name: SharedString,
              local_ae_title: SharedString,
              dimse_host: SharedString,
              dimse_port: SharedString,
              remote_ae_title: SharedString,
              dicomweb_url: SharedString,
              auth_token: SharedString| {
            let result = (|| -> LeafResult<()> {
                let dimse_port = if dimse_port.trim().is_empty() {
                    default_dimse_port()
                } else {
                    dimse_port.trim().parse::<u16>().map_err(|_| {
                        LeafError::Config("DIMSE port must be a valid number".into())
                    })?
                };

                let new_settings = BrowserSettings {
                    last_import_path: settings_for_save.borrow().last_import_path.clone(),
                    node_name: node_name.trim().to_string(),
                    local_ae_title: if local_ae_title.trim().is_empty() {
                        default_local_ae_title()
                    } else {
                        local_ae_title.trim().to_string()
                    },
                    dimse_host: dimse_host.trim().to_string(),
                    dimse_port,
                    remote_ae_title: remote_ae_title.trim().to_string(),
                    dicomweb_url: dicomweb_url.trim().to_string(),
                    auth_token: auth_token.trim().to_string(),
                };

                validate_browser_settings(&new_settings)?;
                save_browser_settings(&imagebox_for_save_settings, &new_settings)?;
                *settings_for_save.borrow_mut() = new_settings.clone();
                if let Some(browser) = browser_for_save_settings.upgrade() {
                    apply_browser_settings(&browser, &new_settings);
                    browser.set_settings_visible(false);
                    let nodes = new_settings.pacs_nodes();
                    refresh_browser(
                        &browser,
                        &imagebox_for_save_settings,
                        &nodes,
                        browser.get_network_mode(),
                        &runtime_for_save,
                        BrowserQuery::default(),
                    )?;
                    browser.set_connection_status("Settings saved".into());
                }
                Ok(())
            })();

            if let Err(error) = result {
                if let Some(browser) = browser_for_save_settings.upgrade() {
                    browser.set_connection_status(format!("Settings save failed: {error}").into());
                }
            }
        },
    );

    // Run the application.
    browser.run()?;
    if let Some(geometry) = capture_window_geometry(browser.window()) {
        if let Err(error) =
            save_window_geometry(&imagebox, BROWSER_WINDOW_GEOMETRY_KEY, &geometry)
        {
            info!("Failed to save browser window geometry: {}", error);
        }
    }

    info!("pacsleaf shutting down");
    Ok(())
}

#[derive(Default)]
struct ImportAccumulator {
    studies: HashMap<String, StudyInfo>,
    series: HashMap<String, SeriesInfo>,
    instances_by_series: HashMap<String, Vec<InstanceInfo>>,
}

struct ImportSummary {
    file_count: usize,
    study_count: usize,
}

fn import_from_path(imagebox: &Imagebox, path: &Path) -> LeafResult<ImportSummary> {
    let mut files = Vec::new();
    collect_files(path, &mut files)?;

    let mut accumulator = ImportAccumulator::default();
    let mut imported_files = 0usize;

    for file in files {
        match import_dicom_file(&file) {
            Ok((study, series, instance)) => {
                merge_import_item(&mut accumulator, study, series, instance);
                imported_files += 1;
            }
            Err(error) => {
                info!(
                    "Skipping non-DICOM or unreadable file {}: {}",
                    file.display(),
                    error
                );
            }
        }
    }

    for (study_uid, study) in accumulator.studies.clone() {
        let mut study = study;
        let series_list = accumulator
            .series
            .values()
            .filter(|series| series.study_uid.0 == study_uid)
            .map(|series| {
                let mut series = series.clone();
                series.num_instances = accumulator
                    .instances_by_series
                    .get(&series.series_uid.0)
                    .map(|instances| instances.len() as u32)
                    .unwrap_or(0);
                series
            })
            .collect::<Vec<_>>();
        let instances = series_list
            .iter()
            .flat_map(|series| {
                accumulator
                    .instances_by_series
                    .get(&series.series_uid.0)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();

        study.num_series = series_list.len() as u32;
        study.num_instances = instances.len() as u32;
        if study.modalities.is_empty() {
            let modalities = series_list
                .iter()
                .map(|series| series.modality.clone())
                .filter(|modality| !modality.is_empty())
                .collect::<BTreeSet<_>>();
            study.modalities = modalities.into_iter().collect();
        }

        imagebox.store_study(&study, &series_list, &instances)?;

        // Generate thumbnails for each series from the middle slice
        for series_info in &series_list {
            let series_uid = &series_info.series_uid.0;
            let series_instances = accumulator
                .instances_by_series
                .get(series_uid)
                .cloned()
                .unwrap_or_default();
            if let Err(e) = generate_and_store_thumbnail(imagebox, series_uid, &series_instances) {
                info!("Thumbnail generation failed for {}: {}", series_uid, e);
            }
        }
    }

    Ok(ImportSummary {
        file_count: imported_files,
        study_count: accumulator.studies.len(),
    })
}

fn merge_import_item(
    accumulator: &mut ImportAccumulator,
    study: StudyInfo,
    series: SeriesInfo,
    instance: InstanceInfo,
) {
    accumulator
        .studies
        .entry(study.study_uid.0.clone())
        .and_modify(|existing| {
            if existing.patient.patient_name.trim().is_empty()
                && !study.patient.patient_name.trim().is_empty()
            {
                existing.patient.patient_name = study.patient.patient_name.clone();
            }
            if existing.patient.patient_id.trim().is_empty()
                && !study.patient.patient_id.trim().is_empty()
            {
                existing.patient.patient_id = study.patient.patient_id.clone();
            }
            if existing.study_description.is_none() {
                existing.study_description = study.study_description.clone();
            }
            if existing.accession_number.is_none() {
                existing.accession_number = study.accession_number.clone();
            }
            if existing.study_date.is_none() {
                existing.study_date = study.study_date;
            }
            if existing.modalities.is_empty() {
                existing.modalities = study.modalities.clone();
            }
        })
        .or_insert(study);

    accumulator
        .series
        .entry(series.series_uid.0.clone())
        .and_modify(|existing| {
            if existing.series_description.is_none() {
                existing.series_description = series.series_description.clone();
            }
            if existing.series_number.is_none() {
                existing.series_number = series.series_number;
            }
            if existing.pixel_spacing.is_none() {
                existing.pixel_spacing = series.pixel_spacing;
            }
            if existing.rows.is_none() {
                existing.rows = series.rows;
            }
            if existing.columns.is_none() {
                existing.columns = series.columns;
            }
        })
        .or_insert(series);

    let instances = accumulator
        .instances_by_series
        .entry(instance.series_uid.0.clone())
        .or_default();
    if !instances
        .iter()
        .any(|existing| existing.sop_instance_uid.0 == instance.sop_instance_uid.0)
    {
        instances.push(instance);
    }
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> LeafResult<()> {
    if !path.exists() {
        return Err(LeafError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Path does not exist: {}", path.display()),
        )));
    }
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if child.is_dir() {
            collect_files(&child, files)?;
        } else if child.is_file() {
            files.push(child);
        }
    }
    Ok(())
}

fn cli_import_paths() -> Vec<PathBuf> {
    let mut args = std::env::args_os().skip(1);
    let mut paths = Vec::new();
    while let Some(arg) = args.next() {
        if arg == "--import" {
            if let Some(path) = args.next() {
                paths.push(PathBuf::from(path));
            }
        }
    }
    paths
}

fn open_remote_viewer_for_study(
    imagebox: &Rc<Imagebox>,
    nodes: &[PacsNodeConfig],
    runtime: &tokio::runtime::Runtime,
    remote: &RemoteStudyRef,
) -> LeafResult<leaf_ui::StudyViewerWindow> {
    let node = find_node(nodes, &remote.node_name)?;
    let client = DicomWebClient::new(node)?;
    let (series_list, patient_name, study_description) =
        runtime.block_on(load_remote_series_and_metadata(&client, &remote.study_uid))?;

    let viewer =
        leaf_ui::StudyViewerWindow::new().map_err(|error| LeafError::Render(error.to_string()))?;
    if let Some(geometry) = load_window_geometry(imagebox, VIEWER_WINDOW_GEOMETRY_KEY)? {
        apply_window_geometry(viewer.window(), geometry);
    }
    viewer.set_patient_name(patient_name.into());
    viewer.set_study_description(study_description.into());
    viewer.set_measurement_panel_visible(false);
    viewer.set_volume_preview_active(false);
    viewer.set_measurements(ModelRc::from(Rc::new(VecModel::from(Vec::<
        leaf_ui::MeasurementEntry,
    >::new()))));
    install_viewer_tool_state(&viewer);

    let active_series_uid = series_list
        .first()
        .map(|series| series.series_uid.clone())
        .unwrap_or_default();
    set_remote_series_model(&viewer, &series_list, &active_series_uid);
    let runtime_handle = runtime.handle().clone();
    update_remote_viewer_image(
        &runtime_handle,
        &viewer,
        &client,
        &remote.study_uid,
        &active_series_uid,
    )?;
    let remote_study_uid = remote.study_uid.clone();
    let shared_client = Rc::new(client);
    let shared_series = Rc::new(series_list);

    let viewer_weak = viewer.as_weak();
    let viewer_weak_for_volume = viewer.as_weak();
    let viewer_weak_for_close = viewer.as_weak();
    let series_for_callback = shared_series.clone();
    let client_for_callback = shared_client.clone();
    let study_uid_for_callback = remote_study_uid.clone();
    let imagebox_for_close = imagebox.clone();
    viewer.on_series_selected(move |series_uid| {
        if let Some(viewer) = viewer_weak.upgrade() {
            set_remote_series_model(&viewer, &series_for_callback, series_uid.as_str());
            if let Err(error) = update_remote_viewer_image(
                &runtime_handle,
                &viewer,
                &client_for_callback,
                &study_uid_for_callback,
                series_uid.as_str(),
            ) {
                viewer.set_study_description(format!("Remote open failed: {error}").into());
            }
        }
    });
    viewer.on_toggle_volume_preview(move || {
        if let Some(viewer) = viewer_weak_for_volume.upgrade() {
            viewer.set_connection_status("3D preview is only available for local studies".into());
        }
    });
    viewer.window().on_close_requested(move || {
        if let Some(viewer) = viewer_weak_for_close.upgrade() {
            if let Some(geometry) = capture_window_geometry(viewer.window()) {
                if let Err(error) = save_window_geometry(
                    &imagebox_for_close,
                    VIEWER_WINDOW_GEOMETRY_KEY,
                    &geometry,
                ) {
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

async fn load_remote_series_and_metadata(
    client: &DicomWebClient,
    study_uid: &str,
) -> LeafResult<(Vec<RemoteSeriesRef>, String, String)> {
    let series = client.search_series(study_uid, &[]).await?;
    let series_list = series
        .iter()
        .map(|item| RemoteSeriesRef {
            series_uid: qido_first_string(item, "0020000E").unwrap_or_default(),
            series_number: qido_first_int(item, "00200011"),
            modality: qido_first_string(item, "00080060").unwrap_or_else(|| "-".to_string()),
            description: qido_first_string(item, "0008103E").unwrap_or_else(|| "-".to_string()),
            instance_count: qido_first_int(item, "00201209"),
        })
        .filter(|item| !item.series_uid.is_empty())
        .collect::<Vec<_>>();

    let study_rows = client
        .search_studies(&[("StudyInstanceUID", study_uid)])
        .await?;
    let first_study = study_rows.first();
    let patient_name = first_study
        .and_then(|study| qido_first_string(study, "00100010"))
        .unwrap_or_else(|| format!("Remote study {study_uid}"));
    let study_description = first_study
        .and_then(|study| qido_first_string(study, "00081030"))
        .unwrap_or_else(|| "Remote study".to_string());

    Ok((series_list, patient_name, study_description))
}

fn set_remote_series_model(
    viewer: &leaf_ui::StudyViewerWindow,
    series_list: &[RemoteSeriesRef],
    active_series_uid: &str,
) {
    let entries = series_list
        .iter()
        .map(|series| leaf_ui::SeriesThumbnail {
            series_uid: series.series_uid.clone().into(),
            series_number: series.series_number,
            modality: series.modality.clone().into(),
            description: series.description.clone().into(),
            instance_count: series.instance_count,
            active: series.series_uid == active_series_uid,
            thumbnail: slint::Image::default(),
            has_thumbnail: false,
        })
        .collect::<Vec<_>>();
    viewer.set_series_list(ModelRc::from(Rc::new(VecModel::from(entries))));
}

fn update_remote_viewer_image(
    runtime: &tokio::runtime::Handle,
    viewer: &leaf_ui::StudyViewerWindow,
    client: &DicomWebClient,
    study_uid: &str,
    series_uid: &str,
) -> LeafResult<()> {
    let encoded = runtime.block_on(async {
        let instances = client.search_instances(study_uid, series_uid, &[]).await?;
        let instance_uid = instances
            .first()
            .and_then(|item| qido_first_string(item, "00080018"))
            .ok_or_else(|| LeafError::NoData("Remote series has no instances".into()))?;
        client
            .get_rendered_frame(study_uid, series_uid, &instance_uid, 1)
            .await
    })?;

    let decoded =
        load_from_memory(&encoded).map_err(|error| LeafError::Render(error.to_string()))?;
    let rgba = decoded.into_rgba8();
    viewer.set_viewport_image(
        leaf_ui::image_from_rgba8(rgba.width(), rgba.height(), rgba.into_raw())
            .map_err(|error| LeafError::Render(error.to_string()))?,
    );
    viewer.set_volume_preview_active(false);
    viewer.set_window_info("Remote render".into());
    viewer.set_slice_info("1/1".into());
    viewer.set_zoom_info("Fit".into());
    Ok(())
}

const THUMB_SIZE: usize = 64;

/// Generate a 64×64 RGBA thumbnail from the middle slice of a series and store it in the DB.
fn generate_and_store_thumbnail(
    imagebox: &Imagebox,
    series_uid: &str,
    instances: &[InstanceInfo],
) -> LeafResult<()> {
    if instances.is_empty() {
        return Ok(());
    }

    // Pick the middle instance
    let mid = instances.len() / 2;
    let instance = &instances[mid];
    let file_path = instance
        .file_path
        .as_ref()
        .ok_or_else(|| LeafError::NoData("Instance has no file path".into()))?;

    let frame = decode_frame(Path::new(file_path), 0)?;
    if frame.width == 0 || frame.height == 0 {
        return Ok(());
    }

    // Downscale to THUMB_SIZE × THUMB_SIZE using nearest-neighbor (fast)
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

    imagebox.store_thumbnail(series_uid, &rgba)?;
    Ok(())
}
