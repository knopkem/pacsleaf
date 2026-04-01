//! Pixel data extraction and decoding from DICOM files.

use dicom_toolkit_data::FileFormat;
use dicom_toolkit_image::DicomImage;
use leaf_core::error::{LeafError, LeafResult};
use std::path::Path;

/// Decoded image frame as pixel data.
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    /// Pixel data (grayscale or RGB depending on output_channels).
    pub pixels: Vec<u8>,
    /// Number of channels per pixel (1 = grayscale, 3 = RGB).
    pub channels: u8,
    pub window_center: f64,
    pub window_width: f64,
}

/// Decode a single frame from a DICOM file.
pub fn decode_frame(path: &Path, frame_index: u32) -> LeafResult<DecodedFrame> {
    decode_frame_with_window(path, frame_index, None)
}

/// Decode a single frame with an optional window/level override.
pub fn decode_frame_with_window(
    path: &Path,
    frame_index: u32,
    window_override: Option<(f64, f64)>,
) -> LeafResult<DecodedFrame> {
    let file = FileFormat::open(path).map_err(|e| LeafError::DicomParse(e.to_string()))?;
    let mut image =
        DicomImage::from_dataset(&file.dataset).map_err(|e| LeafError::DicomParse(e.to_string()))?;

    if let Some((center, width)) = window_override {
        image
            .set_window(center, width)
            .map_err(|e| LeafError::DicomParse(e.to_string()))?;
    } else if image.window_center.is_none() || image.window_width.is_none() {
        image.auto_window();
    }

    let pixels = image
        .frame_u8(frame_index)
        .map_err(|e| LeafError::DicomParse(e.to_string()))?;

    Ok(DecodedFrame {
        width: image.columns,
        height: image.rows,
        pixels,
        channels: image.output_channels(),
        window_center: image.window_center.unwrap_or(0.0),
        window_width: image.window_width.unwrap_or(0.0),
    })
}

/// Get the number of frames in a DICOM file.
pub fn frame_count(path: &Path) -> LeafResult<usize> {
    let file = FileFormat::open(path).map_err(|e| LeafError::DicomParse(e.to_string()))?;
    let ds = &file.dataset;
    let count = ds
        .get_string(dicom_toolkit_dict::tags::NUMBER_OF_FRAMES)
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(1);
    Ok(count)
}
