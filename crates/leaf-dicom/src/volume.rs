//! Volume assembly from a DICOM series for use with volren-rs.

use dicom_toolkit_data::FileFormat;
use dicom_toolkit_dict::{tags, Tag};
use glam::{DMat3, DVec3};
use leaf_core::domain::SeriesUid;
use leaf_core::error::{LeafError, LeafResult};
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
    rows: u16,
    columns: u16,
}

/// Assemble a 3D volume from a sorted list of DICOM file paths.
///
/// Files must belong to the same series and have consistent geometry.
pub fn assemble_volume(
    file_paths: &[String],
    _series_uid: &SeriesUid,
) -> LeafResult<Volume<i16>> {
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

        entries.push(SliceEntry {
            path: path.clone(),
            instance_number,
            image_position,
            rows,
            columns,
        });
    }

    // Step 2: Sort by image position (Z-component) or instance number.
    entries.sort_by(|a, b| {
        if let (Some(pa), Some(pb)) = (&a.image_position, &b.image_position) {
            pa.z.partial_cmp(&pb.z).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.instance_number.cmp(&b.instance_number)
        }
    });

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
    let origin = entries[0]
        .image_position
        .unwrap_or(DVec3::ZERO);
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

fn extract_spacing(
    ds: &dicom_toolkit_data::DataSet,
    entries: &[SliceEntry],
) -> DVec3 {
    let pixel_spacing = ds
        .get_string(PIXEL_SPACING)
        .and_then(|s| {
            let parts: Vec<f64> = s.split('\\').filter_map(|p| p.trim().parse().ok()).collect();
            if parts.len() >= 2 {
                Some((parts[0], parts[1]))
            } else {
                None
            }
        })
        .unwrap_or((1.0, 1.0));

    // Compute slice spacing from positions or use SliceThickness.
    let slice_spacing = if entries.len() >= 2 {
        if let (Some(p0), Some(p1)) = (&entries[0].image_position, &entries[1].image_position) {
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
    ds.get_string(tags::IMAGE_ORIENTATION_PATIENT)
        .and_then(|s| {
            let parts: Vec<f64> = s.split('\\').filter_map(|p| p.trim().parse().ok()).collect();
            if parts.len() >= 6 {
                let row = DVec3::new(parts[0], parts[1], parts[2]);
                let col = DVec3::new(parts[3], parts[4], parts[5]);
                let normal = row.cross(col).normalize();
                Some(DMat3::from_cols(row, col, normal))
            } else {
                None
            }
        })
        .unwrap_or(DMat3::IDENTITY)
}
