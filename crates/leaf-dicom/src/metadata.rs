//! Extract domain model types from DICOM datasets.

use dicom_toolkit_data::{DataSet, FileFormat};
use dicom_toolkit_dict::{tags, Tag};
use leaf_core::domain::*;
use leaf_core::error::{LeafError, LeafResult};
use std::path::Path;

// Tags not yet defined in dicom-toolkit-dict constants.
const BODY_PART_EXAMINED: Tag = Tag::new(0x0018, 0x0015);
const SLICE_THICKNESS: Tag = Tag::new(0x0018, 0x0050);
const PIXEL_SPACING: Tag = Tag::new(0x0028, 0x0030);

/// Extract study-level information from a DICOM dataset.
pub fn extract_study_info(ds: &DataSet) -> LeafResult<StudyInfo> {
    let study_uid = ds
        .get_string(tags::STUDY_INSTANCE_UID)
        .ok_or_else(|| LeafError::DicomParse("Missing StudyInstanceUID".into()))?
        .to_string();

    let patient = PatientInfo {
        patient_id: ds.get_string(tags::PATIENT_ID).unwrap_or("").to_string(),
        patient_name: ds.get_string(tags::PATIENT_NAME).unwrap_or("").to_string(),
        birth_date: ds
            .get_string(tags::PATIENT_BIRTH_DATE)
            .and_then(parse_dicom_date),
        sex: ds.get_string(tags::PATIENT_SEX).map(|s| s.to_string()),
    };

    Ok(StudyInfo {
        study_uid: StudyUid(study_uid),
        patient,
        study_date: ds
            .get_string(tags::STUDY_DATE)
            .and_then(parse_dicom_date),
        study_time: ds.get_string(tags::STUDY_TIME).map(|s| s.to_string()),
        study_description: ds.get_string(tags::STUDY_DESCRIPTION).map(|s| s.to_string()),
        accession_number: ds.get_string(tags::ACCESSION_NUMBER).map(|s| s.to_string()),
        referring_physician: ds.get_string(tags::REFERRING_PHYSICIAN_NAME).map(|s| s.to_string()),
        modalities: ds
            .get_string(tags::MODALITY)
            .map(|m| vec![m.to_string()])
            .unwrap_or_default(),
        num_series: 0,
        num_instances: 0,
    })
}

/// Extract series-level information from a DICOM dataset.
pub fn extract_series_info(ds: &DataSet) -> LeafResult<SeriesInfo> {
    let series_uid = ds
        .get_string(tags::SERIES_INSTANCE_UID)
        .ok_or_else(|| LeafError::DicomParse("Missing SeriesInstanceUID".into()))?
        .to_string();
    let study_uid = ds
        .get_string(tags::STUDY_INSTANCE_UID)
        .ok_or_else(|| LeafError::DicomParse("Missing StudyInstanceUID".into()))?
        .to_string();

    Ok(SeriesInfo {
        series_uid: SeriesUid(series_uid),
        study_uid: StudyUid(study_uid),
        series_number: ds.get_i32(tags::SERIES_NUMBER),
        series_description: ds.get_string(tags::SERIES_DESCRIPTION).map(|s| s.to_string()),
        modality: ds.get_string(tags::MODALITY).unwrap_or("").to_string(),
        body_part: ds.get_string(BODY_PART_EXAMINED).map(|s| s.to_string()),
        num_instances: 0,
        rows: ds.get_u16(tags::ROWS),
        columns: ds.get_u16(tags::COLUMNS),
        pixel_spacing: extract_pixel_spacing(ds),
        slice_thickness: ds.get_f64(SLICE_THICKNESS),
    })
}

/// Extract instance-level information from a DICOM dataset.
pub fn extract_instance_info(ds: &DataSet, file_path: Option<String>) -> LeafResult<InstanceInfo> {
    let sop_instance_uid = ds
        .get_string(tags::SOP_INSTANCE_UID)
        .ok_or_else(|| LeafError::DicomParse("Missing SOPInstanceUID".into()))?
        .to_string();
    let series_uid = ds
        .get_string(tags::SERIES_INSTANCE_UID)
        .ok_or_else(|| LeafError::DicomParse("Missing SeriesInstanceUID".into()))?
        .to_string();
    let study_uid = ds
        .get_string(tags::STUDY_INSTANCE_UID)
        .ok_or_else(|| LeafError::DicomParse("Missing StudyInstanceUID".into()))?
        .to_string();

    Ok(InstanceInfo {
        sop_instance_uid: SopInstanceUid(sop_instance_uid),
        series_uid: SeriesUid(series_uid),
        study_uid: StudyUid(study_uid),
        sop_class_uid: ds.get_string(tags::SOP_CLASS_UID).unwrap_or("").to_string(),
        instance_number: ds.get_i32(tags::INSTANCE_NUMBER),
        transfer_syntax_uid: String::new(),
        file_path,
    })
}

/// Open a DICOM Part 10 file and extract all metadata levels.
pub fn import_dicom_file(
    path: &Path,
) -> LeafResult<(StudyInfo, SeriesInfo, InstanceInfo)> {
    let file = FileFormat::open(path).map_err(|e| LeafError::DicomParse(e.to_string()))?;
    let ds = &file.dataset;

    let study = extract_study_info(ds)?;
    let series = extract_series_info(ds)?;
    let instance = extract_instance_info(ds, Some(path.to_string_lossy().into_owned()))?;

    Ok((study, series, instance))
}

fn extract_pixel_spacing(ds: &DataSet) -> Option<(f64, f64)> {
    let spacing_str = ds.get_string(PIXEL_SPACING)?;
    let parts: Vec<&str> = spacing_str.split('\\').collect();
    if parts.len() >= 2 {
        let row: f64 = parts[0].trim().parse().ok()?;
        let col: f64 = parts[1].trim().parse().ok()?;
        Some((row, col))
    } else {
        None
    }
}

fn parse_dicom_date(s: &str) -> Option<chrono::NaiveDate> {
    chrono::NaiveDate::parse_from_str(s.trim(), "%Y%m%d").ok()
}
