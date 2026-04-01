//! Volume assembly from a DICOM series for use with volren-rs.

use dicom_toolkit_data::FileFormat;
use dicom_toolkit_dict::{tags, Tag};
use glam::{DMat3, DVec3};
use leaf_core::domain::SeriesUid;
use leaf_core::error::{LeafError, LeafResult};
use std::cmp::Ordering;
use std::path::Path;
use tracing::debug;
use volren_core::volume::Volume;

// Tags not yet defined in dicom-toolkit-dict constants.
const PIXEL_SPACING: Tag = Tag::new(0x0028, 0x0030);
const SLICE_THICKNESS: Tag = Tag::new(0x0018, 0x0050);

/// Metadata needed to sort slices and build volumes.
struct SliceEntry {
    path: String,
    instance_number: i32,
    image_position: Option<DVec3>,
    image_orientation: Option<(DVec3, DVec3)>,
    rows: u16,
    columns: u16,
}

/// Assemble a 3D volume from a sorted list of DICOM file paths.
///
/// Files must belong to the same series and have consistent geometry.
pub fn assemble_volume(file_paths: &[String], _series_uid: &SeriesUid) -> LeafResult<Volume<i16>> {
    if file_paths.is_empty() {
        return Err(LeafError::VolumeAssembly("No files provided".into()));
    }

    // Step 1: Parse metadata for sorting.
    let mut entries: Vec<SliceEntry> = Vec::with_capacity(file_paths.len());
    for path in file_paths {
        let file =
            FileFormat::open(Path::new(path)).map_err(|e| LeafError::DicomParse(e.to_string()))?;
        let ds = &file.dataset;
        let rows = ds.get_u16(tags::ROWS).unwrap_or(0);
        let columns = ds.get_u16(tags::COLUMNS).unwrap_or(0);
        let instance_number = ds.get_i32(tags::INSTANCE_NUMBER).unwrap_or(0);
        let image_position = extract_position(ds);
        let image_orientation = extract_orientation_vectors(ds);

        entries.push(SliceEntry {
            path: path.clone(),
            instance_number,
            image_position,
            image_orientation,
            rows,
            columns,
        });
    }

    // Step 2: Sort by slice geometry first, then fall back to instance number.
    sort_slices_for_volume(&mut entries);

    let width = entries[0].columns as usize;
    let height = entries[0].rows as usize;
    let depth = entries.len();

    debug!(
        "Assembling volume: {}x{}x{} from {} slices",
        width,
        height,
        depth,
        file_paths.len()
    );

    // Step 3: Extract pixel data for each slice.
    let mut voxels: Vec<i16> = Vec::with_capacity(width * height * depth);
    for entry in &entries {
        let file = FileFormat::open(Path::new(&entry.path))
            .map_err(|e| LeafError::DicomParse(e.to_string()))?;
        let ds = &file.dataset;
        let pixel_data = ds
            .get_bytes(tags::PIXEL_DATA)
            .ok_or_else(|| LeafError::VolumeAssembly("Missing pixel data".into()))?;

        // Interpret as i16 (common for CT).
        let slice_voxels: &[i16] = bytemuck::cast_slice(pixel_data);
        if slice_voxels.len() < width * height {
            return Err(LeafError::VolumeAssembly(format!(
                "Slice has {} voxels, expected {}",
                slice_voxels.len(),
                width * height
            )));
        }
        voxels.extend_from_slice(&slice_voxels[..width * height]);
    }

    // Step 4: Extract geometric info.
    let first_file = FileFormat::open(Path::new(&entries[0].path))
        .map_err(|e| LeafError::DicomParse(e.to_string()))?;
    let first_ds = &first_file.dataset;

    let spacing = extract_spacing(first_ds, &entries);
    let origin = entries[0].image_position.unwrap_or(DVec3::ZERO);
    let direction = extract_orientation(first_ds);

    let volume = Volume::from_data(
        voxels,
        glam::UVec3::new(width as u32, height as u32, depth as u32),
        spacing,
        origin,
        direction,
        1,
    )
    .map_err(|e| LeafError::VolumeAssembly(e.to_string()))?;

    Ok(volume)
}

fn extract_position(ds: &dicom_toolkit_data::DataSet) -> Option<DVec3> {
    let s = ds.get_string(tags::IMAGE_POSITION_PATIENT)?;
    let parts: Vec<f64> = s
        .split('\\')
        .filter_map(|p| p.trim().parse().ok())
        .collect();
    if parts.len() >= 3 {
        Some(DVec3::new(parts[0], parts[1], parts[2]))
    } else {
        None
    }
}

fn extract_spacing(ds: &dicom_toolkit_data::DataSet, entries: &[SliceEntry]) -> DVec3 {
    let pixel_spacing = ds
        .get_string(PIXEL_SPACING)
        .and_then(|s| {
            let parts: Vec<f64> = s
                .split('\\')
                .filter_map(|p| p.trim().parse().ok())
                .collect();
            if parts.len() >= 2 {
                Some((parts[0], parts[1]))
            } else {
                None
            }
        })
        .unwrap_or((1.0, 1.0));

    // Compute slice spacing from positions or use SliceThickness.
    let slice_spacing = if entries.len() >= 2 {
        if let Some(spacing) = projected_slice_spacing(entries) {
            spacing
        } else if let (Some(p0), Some(p1)) =
            (&entries[0].image_position, &entries[1].image_position)
        {
            (p1.z - p0.z).abs()
        } else {
            ds.get_f64(SLICE_THICKNESS).unwrap_or(1.0)
        }
    } else {
        ds.get_f64(SLICE_THICKNESS).unwrap_or(1.0)
    };

    DVec3::new(pixel_spacing.1, pixel_spacing.0, slice_spacing)
}

fn extract_orientation(ds: &dicom_toolkit_data::DataSet) -> DMat3 {
    extract_orientation_vectors(ds)
        .map(|(row, col)| {
            let normal = row.cross(col).normalize_or_zero();
            if normal.length_squared() > 0.0 {
                DMat3::from_cols(row, col, normal)
            } else {
                DMat3::IDENTITY
            }
        })
        .unwrap_or(DMat3::IDENTITY)
}

fn extract_orientation_vectors(ds: &dicom_toolkit_data::DataSet) -> Option<(DVec3, DVec3)> {
    let s = ds.get_string(tags::IMAGE_ORIENTATION_PATIENT)?;
    let parts: Vec<f64> = s
        .split('\\')
        .filter_map(|p| p.trim().parse().ok())
        .collect();
    if parts.len() >= 6 {
        Some((
            DVec3::new(parts[0], parts[1], parts[2]).normalize_or_zero(),
            DVec3::new(parts[3], parts[4], parts[5]).normalize_or_zero(),
        ))
    } else {
        None
    }
}

fn sort_slices_for_volume(entries: &mut [SliceEntry]) {
    if entries.len() <= 1 {
        return;
    }

    let Some(reference_normal) = reference_slice_normal(entries) else {
        entries.sort_by(compare_slice_entries_fallback);
        return;
    };

    let reference_origin = entries
        .iter()
        .find_map(|entry| entry.image_position)
        .unwrap_or(DVec3::ZERO);

    entries.sort_by(|a, b| {
        compare_slice_entries_by_geometry(a, b, reference_origin, reference_normal)
            .then_with(|| compare_slice_entries_fallback(a, b))
    });
}

fn reference_slice_normal(entries: &[SliceEntry]) -> Option<DVec3> {
    let candidate = entries.iter().find_map(|entry| {
        let (row, col) = entry.image_orientation?;
        let normal = row.cross(col).normalize_or_zero();
        (normal.length_squared() > 0.0).then_some(normal)
    })?;
    entries
        .iter()
        .all(|entry| {
            let Some((row, col)) = entry.image_orientation else {
                return false;
            };
            let normal = row.cross(col).normalize_or_zero();
            normal.length_squared() > 0.0 && normal.dot(candidate).abs() > 0.999
        })
        .then_some(candidate)
}

fn compare_slice_entries_by_geometry(
    left: &SliceEntry,
    right: &SliceEntry,
    reference_origin: DVec3,
    reference_normal: DVec3,
) -> Ordering {
    match (left.image_position, right.image_position) {
        (Some(left_pos), Some(right_pos)) => {
            let left_distance = reference_normal.dot(left_pos - reference_origin);
            let right_distance = reference_normal.dot(right_pos - reference_origin);
            left_distance
                .partial_cmp(&right_distance)
                .unwrap_or(Ordering::Equal)
        }
        _ => compare_slice_entries_fallback(left, right),
    }
}

fn compare_slice_entries_fallback(left: &SliceEntry, right: &SliceEntry) -> Ordering {
    left.instance_number
        .cmp(&right.instance_number)
        .then_with(|| left.path.cmp(&right.path))
}

fn projected_slice_spacing(entries: &[SliceEntry]) -> Option<f64> {
    let normal = reference_slice_normal(entries)?;
    let first = entries.first()?.image_position?;
    let second = entries.get(1)?.image_position?;
    Some(normal.dot(second - first).abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice_entry(
        path: &str,
        instance_number: i32,
        image_position: Option<DVec3>,
        image_orientation: Option<(DVec3, DVec3)>,
    ) -> SliceEntry {
        SliceEntry {
            path: path.to_string(),
            instance_number,
            image_position,
            image_orientation,
            rows: 1,
            columns: 1,
        }
    }

    #[test]
    fn volume_sort_prefers_orientation_over_z() {
        let row = DVec3::Y;
        let col = DVec3::Z;
        let mut entries = vec![
            slice_entry("c", 30, Some(DVec3::new(2.0, 0.0, 30.0)), Some((row, col))),
            slice_entry("a", 10, Some(DVec3::new(0.0, 0.0, 50.0)), Some((row, col))),
            slice_entry("b", 20, Some(DVec3::new(1.0, 0.0, 40.0)), Some((row, col))),
        ];

        sort_slices_for_volume(&mut entries);

        let ordered = entries
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ordered, vec!["a", "b", "c"]);
    }

    #[test]
    fn volume_sort_falls_back_to_instance_number() {
        let mut entries = vec![
            slice_entry("c", 3, None, None),
            slice_entry("a", 1, None, None),
            slice_entry("b", 2, None, None),
        ];

        sort_slices_for_volume(&mut entries);

        let ordered = entries
            .iter()
            .map(|entry| entry.instance_number)
            .collect::<Vec<_>>();
        assert_eq!(ordered, vec![1, 2, 3]);
    }
}
