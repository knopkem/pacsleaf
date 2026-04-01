//! Error types for pacsleaf.

use thiserror::Error;

/// Top-level error type used across all pacsleaf crates.
#[derive(Debug, Error)]
pub enum LeafError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("DICOM parse error: {0}")]
    DicomParse(String),

    #[error("DICOM network error: {0}")]
    DicomNetwork(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Rendering error: {0}")]
    Render(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("DICOMweb error: {status} — {message}")]
    DicomWeb { status: u16, message: String },

    #[error("Volume assembly error: {0}")]
    VolumeAssembly(String),

    #[error("No data: {0}")]
    NoData(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("{0}")]
    Other(String),
}

impl From<toml::de::Error> for LeafError {
    fn from(e: toml::de::Error) -> Self {
        LeafError::Config(e.to_string())
    }
}

impl From<toml::ser::Error> for LeafError {
    fn from(e: toml::ser::Error) -> Self {
        LeafError::Config(e.to_string())
    }
}

/// Convenience type alias.
pub type LeafResult<T> = Result<T, LeafError>;
