//! pacsleaf — Pure Rust Medical Imaging Viewer
//!
//! Companion viewer for pacsnode. Provides a fast, modern, dark-themed
//! clinical workstation for radiologists.

use anyhow::Result;
use directories::UserDirs;
use glam::DVec2;
use image::load_from_memory;
use leaf_core::config::{data_dir, PacsNodeConfig, PacsProtocol};
use leaf_core::domain::{InstanceInfo, SeriesInfo, StudyInfo, StudyUid};
use leaf_core::error::{LeafError, LeafResult};
use leaf_db::imagebox::Imagebox;
use leaf_dicom::metadata::{import_dicom_file, read_instance_geometry};
use leaf_dicom::pixel::{decode_frame_with_window, frame_count};
use leaf_net::dicomweb::DicomWebClient;
use leaf_tools::measurement::{Measurement, MeasurementKind, MeasurementValue};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use tracing::info;
use tracing_subscriber::EnvFilter;

struct ViewerSession {
    viewer: slint::Weak<leaf_ui::StudyViewerWindow>,
    series: Vec<SeriesInfo>,
    instances_by_series: std::collections::HashMap<String, Vec<InstanceInfo>>,
    frames_by_series: HashMap<String, Vec<FrameRef>>,
    measurements_by_series: HashMap<String, Vec<Measurement>>,
    active_series_uid: String,
    active_frame_index: usize,
    measurement_panel_visible: bool,
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

struct RemoteStudyRef {
    node_name: String,
    study_uid: String,
}

struct RemoteSeriesRef {
    series_uid: String,
    series_number: i32,
    modality: String,
    description: String,
    instance_count: i32,
}

#[derive(Clone, Copy, Default)]
struct BrowserQuery<'a> {
    patient_name: &'a str,
    patient_id: &'a str,
    accession: &'a str,
}

struct NormalizedBrowserQuery {
    patient_name: String,
    patient_id: String,
    accession: String,
}

impl BrowserQuery<'_> {
    fn normalized(self) -> NormalizedBrowserQuery {
        NormalizedBrowserQuery {
            patient_name: normalize_query(self.patient_name),
            patient_id: normalize_query(self.patient_id),
            accession: normalize_query(self.accession),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BrowserSettings {
    #[serde(default)]
    last_import_path: Option<String>,
    #[serde(default)]
    node_name: String,
    #[serde(default = "default_local_ae_title")]
    local_ae_title: String,
    #[serde(default)]
    dimse_host: String,
    #[serde(default = "default_dimse_port")]
    dimse_port: u16,
    #[serde(default)]
    remote_ae_title: String,
    #[serde(default)]
    dicomweb_url: String,
    #[serde(default)]
    auth_token: String,
}

impl BrowserSettings {
    fn initial_import_directory(&self) -> PathBuf {
        self.last_import_path
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .unwrap_or_else(default_import_directory)
    }

    fn pacs_nodes(&self) -> Vec<PacsNodeConfig> {
        let dimse_ready =
            !self.dimse_host.trim().is_empty() && !self.remote_ae_title.trim().is_empty();
        let dicomweb_url = self.dicomweb_url.trim();
        let dicomweb_ready = !dicomweb_url.is_empty();

        if !dimse_ready && !dicomweb_ready {
            return Vec::new();
        }

        let protocol = match (dimse_ready, dicomweb_ready) {
            (true, true) => PacsProtocol::Both,
            (true, false) => PacsProtocol::Dimse,
            (false, true) => PacsProtocol::Dicomweb,
            (false, false) => unreachable!("handled above"),
        };

        vec![PacsNodeConfig {
            name: if self.node_name.trim().is_empty() {
                "default-node".to_string()
            } else {
                self.node_name.trim().to_string()
            },
            host: self.dimse_host.trim().to_string(),
            port: if dimse_ready { self.dimse_port } else { 0 },
            ae_title: self.remote_ae_title.trim().to_string(),
            protocol,
            dicomweb_url: dicomweb_ready.then(|| dicomweb_url.to_string()),
            auth_token: if self.auth_token.trim().is_empty() {
                None
            } else {
                Some(self.auth_token.trim().to_string())
            },
        }]
    }
}

impl Default for BrowserSettings {
    fn default() -> Self {
        Self {
            last_import_path: None,
            node_name: String::new(),
            local_ae_title: default_local_ae_title(),
            dimse_host: String::new(),
            dimse_port: default_dimse_port(),
            remote_ae_title: String::new(),
            dicomweb_url: String::new(),
            auth_token: String::new(),
        }
    }
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
            BrowserQuery {
                patient_name: name.as_str(),
                patient_id: id.as_str(),
                accession: accession.as_str(),
            },
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

    info!("pacsleaf shutting down");
    Ok(())
}

fn refresh_browser(
    browser: &leaf_ui::StudyBrowserWindow,
    imagebox: &Imagebox,
    nodes: &[PacsNodeConfig],
    include_remote: bool,
    runtime: &tokio::runtime::Runtime,
    query: BrowserQuery<'_>,
) -> LeafResult<()> {
    let query = query.normalized();

    let mut studies = imagebox.list_studies()?;
    studies.sort_by(|a, b| {
        b.study_date
            .cmp(&a.study_date)
            .then_with(|| a.patient.patient_name.cmp(&b.patient.patient_name))
    });

    let mut entries = studies
        .into_iter()
        .filter(|study| {
            matches_study(
                study,
                &query.patient_name,
                &query.patient_id,
                &query.accession,
            )
        })
        .map(|study| {
            let series = imagebox
                .get_series_for_study(&study.study_uid)
                .unwrap_or_default();
            let instance_count = series
                .iter()
                .map(|series| series.num_instances as i32)
                .sum();
            leaf_ui::StudyEntry {
                study_uid: study.study_uid.0.into(),
                patient_name: empty_fallback(&study.patient.patient_name).into(),
                patient_id: empty_fallback(&study.patient.patient_id).into(),
                study_date: study
                    .study_date
                    .map(|date| date.format("%Y-%m-%d").to_string())
                    .unwrap_or_else(|| "-".to_string())
                    .into(),
                modality: study
                    .modalities
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "-".to_string())
                    .into(),
                description: study
                    .study_description
                    .clone()
                    .unwrap_or_else(|| "-".to_string())
                    .into(),
                series_count: series.len() as i32,
                instance_count,
                source: SharedString::from("Local"),
            }
        })
        .collect::<Vec<_>>();

    if include_remote {
        entries.extend(search_remote_studies(runtime, nodes, &query)?);
    }

    let model = Rc::new(VecModel::from(entries));
    browser.set_studies(ModelRc::from(model));
    browser.set_connection_status(connection_status_text(nodes, include_remote).into());
    Ok(())
}

fn search_remote_studies(
    runtime: &tokio::runtime::Runtime,
    nodes: &[PacsNodeConfig],
    query: &NormalizedBrowserQuery,
) -> LeafResult<Vec<leaf_ui::StudyEntry>> {
    if query.patient_name.is_empty() && query.patient_id.is_empty() && query.accession.is_empty() {
        return Ok(Vec::new());
    }

    runtime.block_on(async {
        let mut entries = Vec::new();
        for node in nodes {
            if matches!(node.protocol, PacsProtocol::Dimse) || node.dicomweb_url.is_none() {
                continue;
            }

            let client = DicomWebClient::new(node)?;
            let mut params = Vec::new();
            if !query.patient_name.is_empty() {
                params.push(("PatientName", query.patient_name.as_str()));
            }
            if !query.patient_id.is_empty() {
                params.push(("PatientID", query.patient_id.as_str()));
            }
            if !query.accession.is_empty() {
                params.push(("AccessionNumber", query.accession.as_str()));
            }

            let studies = client.search_studies(&params).await?;
            for study in studies {
                let study_uid = qido_first_string(&study, "0020000D").unwrap_or_default();
                entries.push(leaf_ui::StudyEntry {
                    study_uid: format!("remote:{}:{}", node.name, study_uid).into(),
                    patient_name: qido_first_string(&study, "00100010")
                        .unwrap_or_else(|| "-".to_string())
                        .into(),
                    patient_id: qido_first_string(&study, "00100020")
                        .unwrap_or_else(|| "-".to_string())
                        .into(),
                    study_date: format_qido_date(qido_first_string(&study, "00080020")).into(),
                    modality: qido_first_string(&study, "00080061")
                        .or_else(|| qido_first_string(&study, "00080060"))
                        .unwrap_or_else(|| "-".to_string())
                        .into(),
                    description: qido_first_string(&study, "00081030")
                        .unwrap_or_else(|| "-".to_string())
                        .into(),
                    series_count: qido_first_int(&study, "00201206"),
                    instance_count: qido_first_int(&study, "00201208"),
                    source: format!("Remote: {}", node.name).into(),
                });
            }
        }
        Ok(entries)
    })
}

fn qido_first_string(value: &serde_json::Value, tag: &str) -> Option<String> {
    value
        .get(tag)?
        .get("Value")?
        .as_array()?
        .first()
        .and_then(|entry| match entry {
            serde_json::Value::String(text) => Some(text.clone()),
            serde_json::Value::Object(map) => map
                .get("Alphabetic")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            serde_json::Value::Number(number) => Some(number.to_string()),
            _ => None,
        })
}

fn qido_first_int(value: &serde_json::Value, tag: &str) -> i32 {
    qido_first_string(value, tag)
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(0)
}

fn format_qido_date(value: Option<String>) -> String {
    let Some(value) = value else {
        return "-".to_string();
    };
    if value.len() == 8 {
        format!("{}-{}-{}", &value[0..4], &value[4..6], &value[6..8])
    } else {
        value
    }
}

fn connection_status_text(nodes: &[PacsNodeConfig], include_remote: bool) -> String {
    if !include_remote {
        return "Local imagebox".to_string();
    }
    let dicomweb_count = nodes
        .iter()
        .filter(|node| node.dicomweb_url.is_some())
        .count();
    let dimse_count = nodes
        .iter()
        .filter(|node| {
            !node.host.trim().is_empty() && !node.ae_title.trim().is_empty() && node.port != 0
        })
        .count();

    match (dicomweb_count, dimse_count) {
        (0, 0) => "Network mode: configure a PACS node".to_string(),
        (0, _) => "Network mode: DIMSE configured; browser search uses DICOMweb".to_string(),
        (_, 0) => format!("Local + {dicomweb_count} DICOMweb node(s)"),
        _ => format!("Local + {dicomweb_count} DICOMweb / {dimse_count} DIMSE node(s)"),
    }
}

fn parse_remote_study_ref(value: &str) -> Option<RemoteStudyRef> {
    let (_, rest) = value.split_once("remote:")?;
    let (node_name, study_uid) = rest.split_once(':')?;
    Some(RemoteStudyRef {
        node_name: node_name.to_string(),
        study_uid: study_uid.to_string(),
    })
}

fn find_node<'a>(nodes: &'a [PacsNodeConfig], name: &str) -> LeafResult<&'a PacsNodeConfig> {
    nodes
        .iter()
        .find(|node| node.name == name)
        .ok_or_else(|| LeafError::Config(format!("Unknown PACS node: {name}")))
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
    viewer.set_patient_name(patient_name.into());
    viewer.set_study_description(study_description.into());
    viewer.set_measurement_panel_visible(false);
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
    let series_for_callback = shared_series.clone();
    let client_for_callback = shared_client.clone();
    let study_uid_for_callback = remote_study_uid.clone();
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
    viewer.set_window_info("Remote render".into());
    viewer.set_slice_info("1/1".into());
    viewer.set_zoom_info("Fit".into());
    Ok(())
}

fn open_viewer_for_study(
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
        session.active_series_uid = series_uid.to_string();
        session.active_frame_index = 0;
        session.selected_measurement_id = None;
        session.draft_measurement = None;
        reset_viewport_state(&mut session, true);
        let result = update_series_model(&session)
            .and_then(|_| update_viewer_image(&mut session))
            .and_then(|_| update_measurements_model(&session));
        if let Err(error) = result {
            info!("Failed to switch series: {}", error);
        }
    });

    let session_for_tool = session.clone();
    viewer.on_tool_selected(move |tool| {
        let mut session = session_for_tool.borrow_mut();
        session.active_tool = tool;
        session.drag_state = None;
        session.draft_measurement = None;
        if let Err(error) = apply_viewport_state(&session) {
            info!("Failed to switch tool: {}", error);
        }
    });

    let session_for_scroll = session.clone();
    viewer.on_viewport_scroll(move |delta| {
        let mut session = session_for_scroll.borrow_mut();
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
        session.active_frame_index = 0;
        session.selected_measurement_id = None;
        session.draft_measurement = None;
        reset_viewport_state(&mut session, false);
        if let Err(error) = update_viewer_image(&mut session) {
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

fn install_viewer_tool_state(viewer: &leaf_ui::StudyViewerWindow) {
    let viewer_weak = viewer.as_weak();
    viewer.on_tool_selected(move |tool| {
        if let Some(viewer) = viewer_weak.upgrade() {
            viewer.set_active_tool(tool);
        }
    });
}

fn update_viewer_image(session: &mut ViewerSession) -> LeafResult<()> {
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

fn reset_viewport_state(session: &mut ViewerSession, clear_defaults: bool) {
    session.viewport_scale = 1.0;
    session.viewport_offset_x = 0.0;
    session.viewport_offset_y = 0.0;
    session.drag_state = None;

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

fn load_browser_settings(imagebox: &Imagebox) -> LeafResult<BrowserSettings> {
    imagebox
        .get_setting("browser_settings")?
        .map(|value| {
            serde_json::from_str(&value).map_err(|error| LeafError::Database(error.to_string()))
        })
        .transpose()
        .map(|settings| settings.unwrap_or_default())
}

fn validate_browser_settings(settings: &BrowserSettings) -> LeafResult<()> {
    let has_dimse_host = !settings.dimse_host.trim().is_empty();
    let has_remote_ae = !settings.remote_ae_title.trim().is_empty();

    if has_dimse_host != has_remote_ae {
        return Err(LeafError::Config(
            "DIMSE host and remote AE title must be set together".into(),
        ));
    }

    Ok(())
}

fn save_browser_settings(imagebox: &Imagebox, settings: &BrowserSettings) -> LeafResult<()> {
    let value =
        serde_json::to_string(settings).map_err(|error| LeafError::Database(error.to_string()))?;
    imagebox.set_setting("browser_settings", &value)
}

fn apply_browser_settings(browser: &leaf_ui::StudyBrowserWindow, settings: &BrowserSettings) {
    browser.set_node_name(settings.node_name.clone().into());
    browser.set_local_ae_title(settings.local_ae_title.clone().into());
    browser.set_dimse_host(settings.dimse_host.clone().into());
    browser.set_dimse_port(settings.dimse_port.to_string().into());
    browser.set_remote_ae_title(settings.remote_ae_title.clone().into());
    browser.set_dicomweb_url(settings.dicomweb_url.clone().into());
    browser.set_auth_token(settings.auth_token.clone().into());
}

fn default_local_ae_title() -> String {
    "PACSLEAF".to_string()
}

fn default_dimse_port() -> u16 {
    104
}

fn default_import_directory() -> PathBuf {
    UserDirs::new()
        .map(|dirs| {
            dirs.document_dir()
                .map(|path| path.to_path_buf())
                .unwrap_or_else(|| dirs.home_dir().to_path_buf())
        })
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
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

fn matches_study(study: &StudyInfo, patient_name: &str, patient_id: &str, accession: &str) -> bool {
    contains_case_insensitive(&study.patient.patient_name, patient_name)
        && contains_case_insensitive(&study.patient.patient_id, patient_id)
        && contains_case_insensitive(
            study.accession_number.as_deref().unwrap_or_default(),
            accession,
        )
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    needle.is_empty() || haystack.to_lowercase().contains(needle)
}

fn normalize_query(value: &str) -> String {
    value.trim().to_lowercase()
}

fn empty_fallback(value: &str) -> &str {
    if value.trim().is_empty() {
        "-"
    } else {
        value
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
