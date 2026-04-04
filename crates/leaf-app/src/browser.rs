//! Study browser, search, and browser-backed settings flow for pacsleaf.

use directories::UserDirs;
use leaf_core::config::{PacsNodeConfig, PacsProtocol};
use leaf_core::domain::StudyInfo;
use leaf_core::error::{LeafError, LeafResult};
use leaf_db::imagebox::Imagebox;
use leaf_net::dicomweb::DicomWebClient;
use serde::{Deserialize, Serialize};
use slint::{ModelRc, SharedString, VecModel};
use std::path::PathBuf;
use std::rc::Rc;

pub(crate) const BROWSER_WINDOW_GEOMETRY_KEY: &str = "browser_window_geometry";
pub(crate) const VIEWER_WINDOW_GEOMETRY_KEY: &str = "viewer_window_geometry";

#[derive(Clone, Copy, Default)]
pub(crate) struct BrowserQuery<'a> {
    patient_name: &'a str,
    patient_id: &'a str,
    accession: &'a str,
}

struct NormalizedBrowserQuery {
    patient_name: String,
    patient_id: String,
    accession: String,
}

pub(crate) struct RemoteStudyRef {
    pub(crate) node_name: String,
    pub(crate) study_uid: String,
}

impl<'a> BrowserQuery<'a> {
    pub(crate) const fn new(
        patient_name: &'a str,
        patient_id: &'a str,
        accession: &'a str,
    ) -> Self {
        Self {
            patient_name,
            patient_id,
            accession,
        }
    }

    fn normalized(self) -> NormalizedBrowserQuery {
        NormalizedBrowserQuery {
            patient_name: normalize_query(self.patient_name),
            patient_id: normalize_query(self.patient_id),
            accession: normalize_query(self.accession),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BrowserSettings {
    #[serde(default)]
    pub(crate) last_import_path: Option<String>,
    #[serde(default)]
    pub(crate) node_name: String,
    #[serde(default = "default_local_ae_title")]
    pub(crate) local_ae_title: String,
    #[serde(default)]
    pub(crate) dimse_host: String,
    #[serde(default = "default_dimse_port")]
    pub(crate) dimse_port: u16,
    #[serde(default)]
    pub(crate) remote_ae_title: String,
    #[serde(default)]
    pub(crate) dicomweb_url: String,
    #[serde(default)]
    pub(crate) auth_token: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct WindowGeometry {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl BrowserSettings {
    pub(crate) fn initial_import_directory(&self) -> PathBuf {
        self.last_import_path
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .unwrap_or_else(default_import_directory)
    }

    pub(crate) fn pacs_nodes(&self) -> Vec<PacsNodeConfig> {
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

pub(crate) fn refresh_browser(
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

pub(crate) fn qido_first_string(value: &serde_json::Value, tag: &str) -> Option<String> {
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

pub(crate) fn qido_first_int(value: &serde_json::Value, tag: &str) -> i32 {
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

pub(crate) fn parse_remote_study_ref(value: &str) -> Option<RemoteStudyRef> {
    let (_, rest) = value.split_once("remote:")?;
    let (node_name, study_uid) = rest.split_once(':')?;
    Some(RemoteStudyRef {
        node_name: node_name.to_string(),
        study_uid: study_uid.to_string(),
    })
}

pub(crate) fn find_node<'a>(
    nodes: &'a [PacsNodeConfig],
    name: &str,
) -> LeafResult<&'a PacsNodeConfig> {
    nodes
        .iter()
        .find(|node| node.name == name)
        .ok_or_else(|| LeafError::Config(format!("Unknown PACS node: {name}")))
}

pub(crate) fn load_browser_settings(imagebox: &Imagebox) -> LeafResult<BrowserSettings> {
    imagebox
        .get_setting("browser_settings")?
        .map(|value| {
            serde_json::from_str(&value).map_err(|error| LeafError::Database(error.to_string()))
        })
        .transpose()
        .map(|settings| settings.unwrap_or_default())
}

pub(crate) fn validate_browser_settings(settings: &BrowserSettings) -> LeafResult<()> {
    let has_dimse_host = !settings.dimse_host.trim().is_empty();
    let has_remote_ae = !settings.remote_ae_title.trim().is_empty();

    if has_dimse_host != has_remote_ae {
        return Err(LeafError::Config(
            "DIMSE host and remote AE title must be set together".into(),
        ));
    }

    Ok(())
}

pub(crate) fn save_browser_settings(
    imagebox: &Imagebox,
    settings: &BrowserSettings,
) -> LeafResult<()> {
    let value =
        serde_json::to_string(settings).map_err(|error| LeafError::Database(error.to_string()))?;
    imagebox.set_setting("browser_settings", &value)
}

pub(crate) fn apply_browser_settings(
    browser: &leaf_ui::StudyBrowserWindow,
    settings: &BrowserSettings,
) {
    browser.set_node_name(settings.node_name.clone().into());
    browser.set_local_ae_title(settings.local_ae_title.clone().into());
    browser.set_dimse_host(settings.dimse_host.clone().into());
    browser.set_dimse_port(settings.dimse_port.to_string().into());
    browser.set_remote_ae_title(settings.remote_ae_title.clone().into());
    browser.set_dicomweb_url(settings.dicomweb_url.clone().into());
    browser.set_auth_token(settings.auth_token.clone().into());
}

pub(crate) fn load_window_geometry(
    imagebox: &Imagebox,
    key: &str,
) -> LeafResult<Option<WindowGeometry>> {
    imagebox
        .get_setting(key)?
        .map(|value| {
            serde_json::from_str(&value).map_err(|error| LeafError::Database(error.to_string()))
        })
        .transpose()
}

pub(crate) fn save_window_geometry(
    imagebox: &Imagebox,
    key: &str,
    geometry: &WindowGeometry,
) -> LeafResult<()> {
    let value =
        serde_json::to_string(geometry).map_err(|error| LeafError::Database(error.to_string()))?;
    imagebox.set_setting(key, &value)
}

pub(crate) fn capture_window_geometry(window: &slint::Window) -> Option<WindowGeometry> {
    let position = window.position();
    let size = window.size();
    if size.width == 0 || size.height == 0 {
        return None;
    }
    Some(WindowGeometry {
        x: position.x,
        y: position.y,
        width: size.width,
        height: size.height,
    })
}

pub(crate) fn apply_window_geometry(window: &slint::Window, geometry: WindowGeometry) {
    if geometry.width == 0 || geometry.height == 0 {
        return;
    }
    window.set_size(slint::PhysicalSize::new(geometry.width, geometry.height));
    window.set_position(slint::PhysicalPosition::new(geometry.x, geometry.y));
}

pub(crate) fn default_local_ae_title() -> String {
    "PACSLEAF".to_string()
}

pub(crate) fn default_dimse_port() -> u16 {
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
