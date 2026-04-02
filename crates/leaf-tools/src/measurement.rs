//! Measurement types and computation.

use glam::DVec2;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Source image data used for pixel-based measurement statistics.
#[derive(Debug, Clone, Copy)]
pub struct MeasurementImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [f64],
    pub unit: &'a str,
}

/// A measurement placed on a DICOM image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub id: String,
    pub kind: MeasurementKind,
    pub label: Option<String>,
    /// Series UID this measurement belongs to.
    pub series_uid: String,
    /// Slice index where the measurement was placed.
    pub slice_index: usize,
    /// Timestamp of creation.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// The type and geometry of a measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementKind {
    Line {
        start: DVec2,
        end: DVec2,
    },
    Angle {
        vertex: DVec2,
        arm1: DVec2,
        arm2: DVec2,
    },
    RectangleRoi {
        top_left: DVec2,
        bottom_right: DVec2,
    },
    EllipseRoi {
        center: DVec2,
        radius_x: f64,
        radius_y: f64,
    },
    PolygonRoi {
        points: Vec<DVec2>,
    },
    PixelProbe {
        position: DVec2,
    },
}

/// Computed result of a measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementResult {
    pub measurement_id: String,
    pub value: MeasurementValue,
}

/// The computed value from a measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementValue {
    Distance {
        mm: f64,
    },
    Angle {
        degrees: f64,
    },
    RoiStats {
        mean: f64,
        std_dev: f64,
        min: f64,
        max: f64,
        area_mm2: f64,
        pixel_count: u64,
    },
    PixelValue {
        value: f64,
        unit: String,
    },
}

impl Measurement {
    /// Create a new line measurement.
    pub fn line(series_uid: &str, slice_index: usize, start: DVec2, end: DVec2) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            kind: MeasurementKind::Line { start, end },
            label: None,
            series_uid: series_uid.to_string(),
            slice_index,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a new angle measurement.
    pub fn angle(
        series_uid: &str,
        slice_index: usize,
        vertex: DVec2,
        arm1: DVec2,
        arm2: DVec2,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            kind: MeasurementKind::Angle { vertex, arm1, arm2 },
            label: None,
            series_uid: series_uid.to_string(),
            slice_index,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a new rectangle ROI measurement.
    pub fn rectangle_roi(
        series_uid: &str,
        slice_index: usize,
        top_left: DVec2,
        bottom_right: DVec2,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            kind: MeasurementKind::RectangleRoi {
                top_left,
                bottom_right,
            },
            label: None,
            series_uid: series_uid.to_string(),
            slice_index,
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a new ellipse ROI measurement.
    pub fn ellipse_roi(
        series_uid: &str,
        slice_index: usize,
        center: DVec2,
        radius_x: f64,
        radius_y: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            kind: MeasurementKind::EllipseRoi {
                center,
                radius_x,
                radius_y,
            },
            label: None,
            series_uid: series_uid.to_string(),
            slice_index,
            created_at: chrono::Utc::now(),
        }
    }

    /// Returns the control point positions for this measurement (in image coordinates).
    pub fn handle_positions(&self) -> Vec<DVec2> {
        match &self.kind {
            MeasurementKind::Line { start, end } => vec![*start, *end],
            MeasurementKind::Angle { vertex, arm1, arm2 } => vec![*vertex, *arm1, *arm2],
            MeasurementKind::RectangleRoi {
                top_left,
                bottom_right,
            } => vec![
                *top_left,
                DVec2::new(bottom_right.x, top_left.y),
                *bottom_right,
                DVec2::new(top_left.x, bottom_right.y),
            ],
            MeasurementKind::EllipseRoi {
                center,
                radius_x,
                radius_y,
            } => vec![
                *center,
                DVec2::new(center.x + radius_x, center.y + radius_y),
            ],
            _ => vec![],
        }
    }

    /// Update the position of a specific handle (by index). Returns true if valid.
    pub fn set_handle_position(&mut self, handle_index: usize, pos: DVec2) -> bool {
        match (&mut self.kind, handle_index) {
            (MeasurementKind::Line { start, .. }, 0) => {
                *start = pos;
                true
            }
            (MeasurementKind::Line { end, .. }, 1) => {
                *end = pos;
                true
            }
            (MeasurementKind::Angle { vertex, .. }, 0) => {
                *vertex = pos;
                true
            }
            (MeasurementKind::Angle { arm1, .. }, 1) => {
                *arm1 = pos;
                true
            }
            (MeasurementKind::Angle { arm2, .. }, 2) => {
                *arm2 = pos;
                true
            }
            (
                MeasurementKind::RectangleRoi {
                    top_left,
                    bottom_right,
                },
                idx,
            ) => match idx {
                0 => {
                    *top_left = pos;
                    true
                }
                1 => {
                    top_left.y = pos.y;
                    bottom_right.x = pos.x;
                    true
                }
                2 => {
                    *bottom_right = pos;
                    true
                }
                3 => {
                    top_left.x = pos.x;
                    bottom_right.y = pos.y;
                    true
                }
                _ => false,
            },
            (MeasurementKind::EllipseRoi { center, .. }, 0) => {
                *center = pos;
                true
            }
            (
                MeasurementKind::EllipseRoi {
                    center,
                    radius_x,
                    radius_y,
                },
                1,
            ) => {
                *radius_x = (pos.x - center.x).abs().max(1.0);
                *radius_y = (pos.y - center.y).abs().max(1.0);
                true
            }
            _ => false,
        }
    }

    /// Compute the result of this measurement given pixel spacing.
    pub fn compute(&self, pixel_spacing: (f64, f64)) -> MeasurementResult {
        self.compute_with_image(pixel_spacing, None)
    }

    /// Compute the result of this measurement with optional source pixel data.
    pub fn compute_with_image(
        &self,
        pixel_spacing: (f64, f64),
        image: Option<&MeasurementImage<'_>>,
    ) -> MeasurementResult {
        let value = match &self.kind {
            MeasurementKind::Line { start, end } => {
                let dx = (end.x - start.x) * pixel_spacing.1;
                let dy = (end.y - start.y) * pixel_spacing.0;
                MeasurementValue::Distance {
                    mm: (dx * dx + dy * dy).sqrt(),
                }
            }
            MeasurementKind::Angle { vertex, arm1, arm2 } => {
                let v1 = *arm1 - *vertex;
                let v2 = *arm2 - *vertex;
                let dot = v1.dot(v2);
                let cross = v1.x * v2.y - v1.y * v2.x;
                let angle_rad = cross.atan2(dot).abs();
                MeasurementValue::Angle {
                    degrees: angle_rad.to_degrees(),
                }
            }
            MeasurementKind::RectangleRoi {
                top_left,
                bottom_right,
            } => {
                let w = (bottom_right.x - top_left.x).abs() * pixel_spacing.1;
                let h = (bottom_right.y - top_left.y).abs() * pixel_spacing.0;
                let area_mm2 = w * h;
                image
                    .map(|image| {
                        roi_stats_for_image(image, area_mm2, |point| {
                            point.x >= top_left.x.min(bottom_right.x)
                                && point.x <= top_left.x.max(bottom_right.x)
                                && point.y >= top_left.y.min(bottom_right.y)
                                && point.y <= top_left.y.max(bottom_right.y)
                        })
                    })
                    .unwrap_or_else(|| empty_roi_stats(area_mm2))
            }
            MeasurementKind::EllipseRoi {
                center,
                radius_x,
                radius_y,
            } => {
                let rx_mm = radius_x * pixel_spacing.1;
                let ry_mm = radius_y * pixel_spacing.0;
                let area_mm2 = std::f64::consts::PI * rx_mm * ry_mm;
                image
                    .map(|image| {
                        roi_stats_for_image(image, area_mm2, |point| {
                            let dx = (point.x - center.x) / radius_x.max(1.0e-6);
                            let dy = (point.y - center.y) / radius_y.max(1.0e-6);
                            dx * dx + dy * dy <= 1.0
                        })
                    })
                    .unwrap_or_else(|| empty_roi_stats(area_mm2))
            }
            MeasurementKind::PolygonRoi { points } => {
                // Shoelace formula for area.
                let mut area = 0.0;
                let n = points.len();
                for i in 0..n {
                    let j = (i + 1) % n;
                    area += points[i].x * points[j].y;
                    area -= points[j].x * points[i].y;
                }
                area = area.abs() / 2.0 * pixel_spacing.0 * pixel_spacing.1;
                image
                    .filter(|_| points.len() >= 3)
                    .map(|image| {
                        roi_stats_for_image(image, area, |point| point_in_polygon(point, points))
                    })
                    .unwrap_or_else(|| empty_roi_stats(area))
            }
            MeasurementKind::PixelProbe { position } => image
                .and_then(|image| sample_image_pixel(image, *position))
                .map(|value| MeasurementValue::PixelValue {
                    value,
                    unit: image
                        .map(|image| image.unit)
                        .unwrap_or_default()
                        .to_string(),
                })
                .unwrap_or_else(|| MeasurementValue::PixelValue {
                    value: 0.0,
                    unit: "HU".to_string(),
                }),
        };

        MeasurementResult {
            measurement_id: self.id.clone(),
            value,
        }
    }
}

fn empty_roi_stats(area_mm2: f64) -> MeasurementValue {
    MeasurementValue::RoiStats {
        mean: 0.0,
        std_dev: 0.0,
        min: 0.0,
        max: 0.0,
        area_mm2,
        pixel_count: 0,
    }
}

fn roi_stats_for_image(
    image: &MeasurementImage<'_>,
    area_mm2: f64,
    contains: impl Fn(DVec2) -> bool,
) -> MeasurementValue {
    let width = image.width as usize;
    let height = image.height as usize;
    if width == 0 || height == 0 || image.pixels.len() < width * height {
        return empty_roi_stats(area_mm2);
    }

    let mut pixel_count = 0u64;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for y in 0..height {
        for x in 0..width {
            let point = DVec2::new(x as f64 + 0.5, y as f64 + 0.5);
            if !contains(point) {
                continue;
            }
            let value = image.pixels[y * width + x];
            pixel_count += 1;
            sum += value;
            sum_sq += value * value;
            min = min.min(value);
            max = max.max(value);
        }
    }

    if pixel_count == 0 {
        return empty_roi_stats(area_mm2);
    }

    let mean = sum / pixel_count as f64;
    let variance = (sum_sq / pixel_count as f64) - mean * mean;
    MeasurementValue::RoiStats {
        mean,
        std_dev: variance.max(0.0).sqrt(),
        min,
        max,
        area_mm2,
        pixel_count,
    }
}

fn point_in_polygon(point: DVec2, points: &[DVec2]) -> bool {
    let mut inside = false;
    let mut previous = *points.last().unwrap_or(&point);
    for &current in points {
        let intersects = ((current.y > point.y) != (previous.y > point.y))
            && (point.x
                < (previous.x - current.x) * (point.y - current.y)
                    / (previous.y - current.y).max(f64::EPSILON)
                    + current.x);
        if intersects {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn sample_image_pixel(image: &MeasurementImage<'_>, position: DVec2) -> Option<f64> {
    let x = position.x.floor();
    let y = position.y.floor();
    if x < 0.0 || y < 0.0 || x >= image.width as f64 || y >= image.height as f64 {
        return None;
    }
    let index = y as usize * image.width as usize + x as usize;
    image.pixels.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangle_roi_computes_pixel_statistics() {
        let measurement =
            Measurement::rectangle_roi("series", 0, DVec2::new(1.0, 1.0), DVec2::new(3.0, 3.0));
        let pixels = vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0, //
            12.0, 13.0, 14.0, 15.0,
        ];
        let image = MeasurementImage {
            width: 4,
            height: 4,
            pixels: &pixels,
            unit: "HU",
        };

        let result = measurement.compute_with_image((1.0, 1.0), Some(&image));
        let MeasurementValue::RoiStats {
            mean,
            std_dev,
            min,
            max,
            area_mm2,
            pixel_count,
        } = result.value
        else {
            panic!("expected ROI stats");
        };

        assert!((mean - 7.5).abs() < 1.0e-6);
        assert!((std_dev - 2.0615528128).abs() < 1.0e-6);
        assert_eq!(min, 5.0);
        assert_eq!(max, 10.0);
        assert_eq!(area_mm2, 4.0);
        assert_eq!(pixel_count, 4);
    }

    #[test]
    fn ellipse_roi_counts_pixels_inside_shape() {
        let measurement = Measurement::ellipse_roi("series", 0, DVec2::new(2.0, 2.0), 1.5, 1.5);
        let pixels = vec![
            0.0, 1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, 7.0, //
            8.0, 9.0, 10.0, 11.0, //
            12.0, 13.0, 14.0, 15.0,
        ];
        let image = MeasurementImage {
            width: 4,
            height: 4,
            pixels: &pixels,
            unit: "HU",
        };

        let result = measurement.compute_with_image((1.0, 1.0), Some(&image));
        let MeasurementValue::RoiStats {
            pixel_count,
            min,
            max,
            ..
        } = result.value
        else {
            panic!("expected ROI stats");
        };

        assert_eq!(pixel_count, 4);
        assert_eq!(min, 5.0);
        assert_eq!(max, 10.0);
    }
}
