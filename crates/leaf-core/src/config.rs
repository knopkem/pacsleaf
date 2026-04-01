//! Application settings model and platform defaults.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub database: DatabaseConfig,
    #[serde(default)]
    pub display: DisplayConfig,
    #[serde(default)]
    pub nodes: Vec<PacsNodeConfig>,
}

/// General application settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Local DICOM AE title (max 16 chars).
    #[serde(default = "default_ae_title")]
    pub ae_title: String,
    /// Local Storage SCP port.
    #[serde(default = "default_port")]
    pub port: u16,
    /// Log level (trace, debug, info, warn, error).
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

/// Database and storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to the redb database file.
    pub path: Option<PathBuf>,
    /// Directory for stored DICOM files.
    pub storage_path: Option<PathBuf>,
    /// Maximum decoded frame cache size in MB.
    #[serde(default = "default_cache_size_mb")]
    pub cache_size_mb: u64,
}

/// Display and rendering preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Default color lookup table.
    #[serde(default = "default_lut")]
    pub default_lut: String,
    /// Default interpolation mode.
    #[serde(default = "default_interpolation")]
    pub default_interpolation: String,
    /// Default viewport layout (e.g., "1x1", "2x2").
    #[serde(default = "default_layout")]
    pub default_layout: String,
}

/// Remote PACS node definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacsNodeConfig {
    /// Display name for this node.
    pub name: String,
    /// Hostname or IP address.
    pub host: String,
    /// Port number.
    pub port: u16,
    /// Remote AE title.
    pub ae_title: String,
    /// Protocol to use for this node.
    #[serde(default)]
    pub protocol: PacsProtocol,
    /// Base URL for DICOMweb endpoints (required if protocol is DICOMweb).
    pub dicomweb_url: Option<String>,
    /// Optional bearer token for DICOMweb authentication.
    pub auth_token: Option<String>,
}

/// Communication protocol for a PACS node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PacsProtocol {
    #[default]
    Dimse,
    Dicomweb,
    Both,
}

impl AppConfig {
    /// Load configuration from disk, falling back to defaults.
    pub fn load(path: &std::path::Path) -> Result<Self, crate::error::LeafError> {
        if path.exists() {
            let contents = std::fs::read_to_string(path)?;
            let config: AppConfig = toml::from_str(&contents)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    /// Save the current configuration to disk.
    pub fn save(&self, path: &std::path::Path) -> Result<(), crate::error::LeafError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Resolve the database path using platform-appropriate defaults.
    pub fn resolved_db_path(&self) -> PathBuf {
        self.database
            .path
            .clone()
            .unwrap_or_else(|| data_dir().join("imagebox.redb"))
    }

    /// Resolve the DICOM file storage path.
    pub fn resolved_storage_path(&self) -> PathBuf {
        self.database
            .storage_path
            .clone()
            .unwrap_or_else(|| data_dir().join("storage"))
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            database: DatabaseConfig::default(),
            display: DisplayConfig::default(),
            nodes: Vec::new(),
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            ae_title: default_ae_title(),
            port: default_port(),
            log_level: default_log_level(),
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: None,
            storage_path: None,
            cache_size_mb: default_cache_size_mb(),
        }
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            default_lut: default_lut(),
            default_interpolation: default_interpolation(),
            default_layout: default_layout(),
        }
    }
}

/// Platform-appropriate data directory for pacsleaf.
pub fn data_dir() -> PathBuf {
    directories::ProjectDirs::from("com", "pacsleaf", "pacsleaf")
        .map(|dirs| dirs.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".pacsleaf"))
}

/// Platform-appropriate config directory for pacsleaf.
pub fn config_dir() -> PathBuf {
    directories::ProjectDirs::from("com", "pacsleaf", "pacsleaf")
        .map(|dirs| dirs.config_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".pacsleaf"))
}

fn default_ae_title() -> String {
    "PACSLEAF".to_string()
}
fn default_port() -> u16 {
    11114
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_cache_size_mb() -> u64 {
    512
}
fn default_lut() -> String {
    "grayscale".to_string()
}
fn default_interpolation() -> String {
    "bilinear".to_string()
}
fn default_layout() -> String {
    "1x1".to_string()
}
