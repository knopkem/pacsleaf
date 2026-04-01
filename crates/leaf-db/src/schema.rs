//! Database schema definitions for redb tables.

use redb::TableDefinition;

/// Studies table: StudyUID → JSON-serialized StudyInfo.
pub const STUDIES: TableDefinition<&str, &str> = TableDefinition::new("studies");

/// Series table: SeriesUID → JSON-serialized SeriesInfo.
pub const SERIES: TableDefinition<&str, &str> = TableDefinition::new("series");

/// Instances table: SOPInstanceUID → JSON-serialized InstanceInfo.
pub const INSTANCES: TableDefinition<&str, &str> = TableDefinition::new("instances");

/// Index: PatientID → list of StudyUIDs (JSON array).
pub const PATIENT_INDEX: TableDefinition<&str, &str> = TableDefinition::new("idx_patient");

/// Index: StudyUID → list of SeriesUIDs (JSON array).
pub const STUDY_SERIES_INDEX: TableDefinition<&str, &str> =
    TableDefinition::new("idx_study_series");

/// Index: SeriesUID → list of SOPInstanceUIDs (JSON array).
pub const SERIES_INSTANCE_INDEX: TableDefinition<&str, &str> =
    TableDefinition::new("idx_series_instances");

/// Settings table: key → value (string).
pub const SETTINGS: TableDefinition<&str, &str> = TableDefinition::new("settings");
