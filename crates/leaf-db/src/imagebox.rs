//! Imagebox repository — high-level operations on the local study database.

use crate::schema::*;
use leaf_core::domain::{InstanceInfo, SeriesInfo, SeriesUid, StudyInfo, StudyUid};
use leaf_core::error::{LeafError, LeafResult};
use redb::{Database, ReadableTable};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};

fn db_err(e: impl std::fmt::Display) -> LeafError {
    LeafError::Database(e.to_string())
}

/// The local imagebox database, managing DICOM studies on disk.
pub struct Imagebox {
    db: Arc<Database>,
}

impl Imagebox {
    /// Open or create the imagebox database at the given path.
    pub fn open(path: &Path) -> LeafResult<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(db_err)?;
        }
        let db = Database::create(path).map_err(db_err)?;

        let instance = Self { db: Arc::new(db) };
        instance.ensure_tables()?;
        info!("Imagebox opened at {}", path.display());
        Ok(instance)
    }

    fn ensure_tables(&self) -> LeafResult<()> {
        let txn = self.db.begin_write().map_err(db_err)?;
        txn.open_table(STUDIES).map_err(db_err)?;
        txn.open_table(SERIES).map_err(db_err)?;
        txn.open_table(INSTANCES).map_err(db_err)?;
        txn.open_table(PATIENT_INDEX).map_err(db_err)?;
        txn.open_table(STUDY_SERIES_INDEX).map_err(db_err)?;
        txn.open_table(SERIES_INSTANCE_INDEX).map_err(db_err)?;
        txn.open_table(SETTINGS).map_err(db_err)?;
        txn.commit().map_err(db_err)?;
        Ok(())
    }

    /// Store a study with its series and instances.
    pub fn store_study(
        &self,
        study: &StudyInfo,
        series_list: &[SeriesInfo],
        instances: &[InstanceInfo],
    ) -> LeafResult<()> {
        let txn = self.db.begin_write().map_err(db_err)?;

        {
            let mut studies_table = txn.open_table(STUDIES).map_err(db_err)?;
            let study_json = serde_json::to_string(study).map_err(db_err)?;
            studies_table
                .insert(study.study_uid.0.as_str(), study_json.as_str())
                .map_err(db_err)?;
        }

        {
            let mut series_table = txn.open_table(SERIES).map_err(db_err)?;
            let mut study_series_idx = txn.open_table(STUDY_SERIES_INDEX).map_err(db_err)?;

            let series_uids: Vec<&str> =
                series_list.iter().map(|s| s.series_uid.0.as_str()).collect();
            let idx_json = serde_json::to_string(&series_uids).map_err(db_err)?;
            study_series_idx
                .insert(study.study_uid.0.as_str(), idx_json.as_str())
                .map_err(db_err)?;

            for s in series_list {
                let json = serde_json::to_string(s).map_err(db_err)?;
                series_table
                    .insert(s.series_uid.0.as_str(), json.as_str())
                    .map_err(db_err)?;
            }
        }

        {
            let mut instances_table = txn.open_table(INSTANCES).map_err(db_err)?;
            let mut series_inst_idx = txn.open_table(SERIES_INSTANCE_INDEX).map_err(db_err)?;

            let mut by_series: std::collections::HashMap<&str, Vec<&str>> =
                std::collections::HashMap::new();
            for inst in instances {
                by_series
                    .entry(inst.series_uid.0.as_str())
                    .or_default()
                    .push(inst.sop_instance_uid.0.as_str());
                let json = serde_json::to_string(inst).map_err(db_err)?;
                instances_table
                    .insert(inst.sop_instance_uid.0.as_str(), json.as_str())
                    .map_err(db_err)?;
            }
            for (series_uid, inst_uids) in &by_series {
                let idx_json = serde_json::to_string(inst_uids).map_err(db_err)?;
                series_inst_idx
                    .insert(*series_uid, idx_json.as_str())
                    .map_err(db_err)?;
            }
        }

        // Update patient index.
        {
            let mut patient_idx = txn.open_table(PATIENT_INDEX).map_err(db_err)?;
            let pid = study.patient.patient_id.as_str();
            let mut uids: Vec<String> = match patient_idx.get(pid).map_err(db_err)? {
                Some(val) => serde_json::from_str(val.value()).unwrap_or_default(),
                None => Vec::new(),
            };
            if !uids.iter().any(|u| u == &study.study_uid.0) {
                uids.push(study.study_uid.0.clone());
            }
            let idx_json = serde_json::to_string(&uids).map_err(db_err)?;
            patient_idx
                .insert(pid, idx_json.as_str())
                .map_err(db_err)?;
        }

        txn.commit().map_err(db_err)?;
        debug!(
            "Stored study {} with {} series",
            study.study_uid,
            series_list.len()
        );
        Ok(())
    }

    /// List all studies in the database.
    pub fn list_studies(&self) -> LeafResult<Vec<StudyInfo>> {
        let txn = self.db.begin_read().map_err(db_err)?;
        let table = txn.open_table(STUDIES).map_err(db_err)?;

        let mut studies = Vec::new();
        for entry in table.range::<&str>(..).map_err(db_err)? {
            let (_key, val) = entry.map_err(db_err)?;
            if let Ok(study) = serde_json::from_str::<StudyInfo>(val.value()) {
                studies.push(study);
            }
        }
        Ok(studies)
    }

    /// Get a study by its UID.
    pub fn get_study(&self, study_uid: &StudyUid) -> LeafResult<Option<StudyInfo>> {
        let txn = self.db.begin_read().map_err(db_err)?;
        let table = txn.open_table(STUDIES).map_err(db_err)?;

        match table.get(study_uid.0.as_str()).map_err(db_err)? {
            Some(val) => {
                let study = serde_json::from_str(val.value()).map_err(db_err)?;
                Ok(Some(study))
            }
            None => Ok(None),
        }
    }

    /// Get all series for a study.
    pub fn get_series_for_study(&self, study_uid: &StudyUid) -> LeafResult<Vec<SeriesInfo>> {
        let txn = self.db.begin_read().map_err(db_err)?;
        let idx_table = txn.open_table(STUDY_SERIES_INDEX).map_err(db_err)?;
        let series_table = txn.open_table(SERIES).map_err(db_err)?;

        let series_uids: Vec<String> = match idx_table
            .get(study_uid.0.as_str())
            .map_err(db_err)?
        {
            Some(val) => serde_json::from_str(val.value()).unwrap_or_default(),
            None => return Ok(Vec::new()),
        };

        let mut result = Vec::with_capacity(series_uids.len());
        for uid in &series_uids {
            if let Some(val) = series_table.get(uid.as_str()).map_err(db_err)? {
                if let Ok(s) = serde_json::from_str::<SeriesInfo>(val.value()) {
                    result.push(s);
                }
            }
        }
        Ok(result)
    }

    /// Get all instances for a series.
    pub fn get_instances_for_series(
        &self,
        series_uid: &SeriesUid,
    ) -> LeafResult<Vec<InstanceInfo>> {
        let txn = self.db.begin_read().map_err(db_err)?;
        let idx_table = txn.open_table(SERIES_INSTANCE_INDEX).map_err(db_err)?;
        let inst_table = txn.open_table(INSTANCES).map_err(db_err)?;

        let inst_uids: Vec<String> = match idx_table
            .get(series_uid.0.as_str())
            .map_err(db_err)?
        {
            Some(val) => serde_json::from_str(val.value()).unwrap_or_default(),
            None => return Ok(Vec::new()),
        };

        let mut result = Vec::with_capacity(inst_uids.len());
        for uid in &inst_uids {
            if let Some(val) = inst_table.get(uid.as_str()).map_err(db_err)? {
                if let Ok(inst) = serde_json::from_str::<InstanceInfo>(val.value()) {
                    result.push(inst);
                }
            }
        }
        Ok(result)
    }

    /// Delete a study and all its series/instances from the database.
    pub fn delete_study(&self, study_uid: &StudyUid) -> LeafResult<()> {
        let series = self.get_series_for_study(study_uid)?;

        let txn = self.db.begin_write().map_err(db_err)?;

        // Collect instance UIDs to delete (read from index before mutating).
        let mut instance_uids_to_delete: Vec<String> = Vec::new();
        {
            let series_inst_idx = txn.open_table(SERIES_INSTANCE_INDEX).map_err(db_err)?;
            for s in &series {
                if let Some(val) = series_inst_idx
                    .get(s.series_uid.0.as_str())
                    .map_err(db_err)?
                {
                    let uids: Vec<String> =
                        serde_json::from_str(val.value()).unwrap_or_default();
                    instance_uids_to_delete.extend(uids);
                }
            }
        }

        // Delete instances.
        {
            let mut inst_table = txn.open_table(INSTANCES).map_err(db_err)?;
            for uid in &instance_uids_to_delete {
                let _ = inst_table.remove(uid.as_str());
            }
        }

        // Delete series instance indices and series.
        {
            let mut series_inst_idx = txn.open_table(SERIES_INSTANCE_INDEX).map_err(db_err)?;
            for s in &series {
                let _ = series_inst_idx.remove(s.series_uid.0.as_str());
            }
        }
        {
            let mut series_table = txn.open_table(SERIES).map_err(db_err)?;
            for s in &series {
                let _ = series_table.remove(s.series_uid.0.as_str());
            }
        }

        // Delete study.
        {
            let mut studies_table = txn.open_table(STUDIES).map_err(db_err)?;
            let _ = studies_table.remove(study_uid.0.as_str());
        }
        {
            let mut study_series_idx = txn.open_table(STUDY_SERIES_INDEX).map_err(db_err)?;
            let _ = study_series_idx.remove(study_uid.0.as_str());
        }

        txn.commit().map_err(db_err)?;
        info!("Deleted study {}", study_uid);
        Ok(())
    }
}
