//! DICOM overlay plane extraction helpers.

use dicom_toolkit_data::FileFormat;
use dicom_toolkit_image::Overlay;
use leaf_core::error::{LeafError, LeafResult};
use std::path::Path;

/// Unpacked overlay bitmap ready for compositing onto an image frame.
#[derive(Debug, Clone)]
pub struct OverlayBitmap {
    pub rows: u16,
    pub columns: u16,
    pub origin: (i16, i16),
    pub bitmap: Vec<u8>,
}

/// Load and unpack all overlay planes from a DICOM instance.
pub fn load_overlays(path: &Path) -> LeafResult<Vec<OverlayBitmap>> {
    let file = FileFormat::open(path).map_err(|error| LeafError::DicomParse(error.to_string()))?;
    Ok(Overlay::from_dataset(&file.dataset)
        .into_iter()
        .map(|overlay| OverlayBitmap {
            rows: overlay.rows,
            columns: overlay.columns,
            origin: overlay.origin,
            bitmap: overlay.to_bitmap(),
        })
        .collect())
}
