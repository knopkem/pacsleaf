//! Domain model types for DICOM studies, series, and instances.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// Unique identifier for a DICOM study.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StudyUid(pub String);

/// Unique identifier for a DICOM series.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SeriesUid(pub String);

/// Unique identifier for a DICOM SOP instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SopInstanceUid(pub String);

/// Patient-level information extracted from DICOM metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientInfo {
    pub patient_id: String,
    pub patient_name: String,
    pub birth_date: Option<NaiveDate>,
    pub sex: Option<String>,
}

/// Study-level information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyInfo {
    pub study_uid: StudyUid,
    pub patient: PatientInfo,
    pub study_date: Option<NaiveDate>,
    pub study_time: Option<String>,
    pub study_description: Option<String>,
    pub accession_number: Option<String>,
    pub referring_physician: Option<String>,
    pub modalities: Vec<String>,
    pub num_series: u32,
    pub num_instances: u32,
}

/// Series-level information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesInfo {
    pub series_uid: SeriesUid,
    pub study_uid: StudyUid,
    pub series_number: Option<i32>,
    pub series_description: Option<String>,
    pub modality: String,
    pub body_part: Option<String>,
    pub num_instances: u32,
    pub rows: Option<u16>,
    pub columns: Option<u16>,
    pub pixel_spacing: Option<(f64, f64)>,
    pub slice_thickness: Option<f64>,
}

/// Instance-level information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub sop_instance_uid: SopInstanceUid,
    pub series_uid: SeriesUid,
    pub study_uid: StudyUid,
    pub sop_class_uid: String,
    pub instance_number: Option<i32>,
    #[serde(default)]
    pub image_position_patient: Option<[f64; 3]>,
    #[serde(default)]
    pub image_orientation_patient: Option<[f64; 6]>,
    pub transfer_syntax_uid: String,
    pub file_path: Option<String>,
}

/// Availability status of a study in the local imagebox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyStatus {
    /// Only metadata indexed, files on disk.
    Indexed,
    /// Retrieval from remote PACS in progress.
    Retrieving,
    /// Fully available locally.
    Available,
    /// Retrieval or import failed.
    Failed,
}

/// Source of a study (local or remote PACS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StudySource {
    Local,
    Remote { node_name: String },
}

impl std::fmt::Display for StudyUid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Display for SeriesUid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Display for SopInstanceUid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
