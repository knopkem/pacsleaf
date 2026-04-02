//! Pixel data extraction and decoding from DICOM files.

use dicom_toolkit_data::FileFormat;
use dicom_toolkit_dict::tags;
use dicom_toolkit_image::{pixel, DicomImage, ModalityLut, PixelRepresentation};
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

/// Decoded measurement frame with modality-space grayscale values when available.
pub struct MeasurementFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f64>,
    pub unit: String,
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
    let mut image = DicomImage::from_dataset(&file.dataset)
        .map_err(|e| LeafError::DicomParse(e.to_string()))?;

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

/// Decode a frame for measurements using modality-space values when possible.
pub fn decode_frame_for_measurements(
    path: &Path,
    frame_index: u32,
) -> LeafResult<MeasurementFrame> {
    let file = FileFormat::open(path).map_err(|e| LeafError::DicomParse(e.to_string()))?;
    let ds = &file.dataset;
    let image = DicomImage::from_dataset(ds).map_err(|e| LeafError::DicomParse(e.to_string()))?;

    let pixels = if image.photometric.is_grayscale() {
        let raw = image
            .frame_bytes(frame_index)
            .map_err(|e| LeafError::DicomParse(e.to_string()))?;
        let modality_lut = ModalityLut::new(image.rescale_intercept, image.rescale_slope);
        match (image.bits_allocated, image.pixel_representation) {
            (8, _) => modality_lut.apply_to_frame_u8(raw),
            (16, PixelRepresentation::Unsigned) => {
                let pixels = pixel::decode_u16_le(raw);
                let pixels = pixel::mask_u16(&pixels, image.bits_stored, image.high_bit);
                modality_lut.apply_to_frame_u16(&pixels)
            }
            (16, PixelRepresentation::Signed) => {
                let pixels = pixel::decode_i16_le(raw);
                let pixels = pixel::mask_i16(&pixels, image.bits_stored, image.high_bit);
                modality_lut.apply_to_frame_i16(&pixels)
            }
            _ => {
                return Err(LeafError::DicomParse(format!(
                    "Unsupported BitsAllocated={} for ROI statistics",
                    image.bits_allocated
                )))
            }
        }
    } else {
        let frame = image
            .frame_u8(frame_index)
            .map_err(|e| LeafError::DicomParse(e.to_string()))?;
        match image.output_channels() {
            3 => frame
                .chunks_exact(3)
                .map(|rgb| {
                    0.2126 * f64::from(rgb[0])
                        + 0.7152 * f64::from(rgb[1])
                        + 0.0722 * f64::from(rgb[2])
                })
                .collect(),
            1 => frame.into_iter().map(f64::from).collect(),
            channels => {
                return Err(LeafError::DicomParse(format!(
                    "Unsupported channel count {channels} for ROI statistics"
                )))
            }
        }
    };

    let unit = if image.photometric.is_grayscale() && ds.get_string(tags::MODALITY) == Some("CT") {
        "HU".to_string()
    } else {
        String::new()
    };

    Ok(MeasurementFrame {
        width: image.columns,
        height: image.rows,
        pixels,
        unit,
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
