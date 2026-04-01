//! Measurement types and computation.

use glam::DVec2;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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

    /// Compute the result of this measurement given pixel spacing.
    pub fn compute(&self, pixel_spacing: (f64, f64)) -> MeasurementResult {
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
                MeasurementValue::RoiStats {
                    mean: 0.0,
                    std_dev: 0.0,
                    min: 0.0,
                    max: 0.0,
                    area_mm2: w * h,
                    pixel_count: 0,
                }
            }
            MeasurementKind::EllipseRoi {
                radius_x, radius_y, ..
            } => {
                let rx_mm = radius_x * pixel_spacing.1;
                let ry_mm = radius_y * pixel_spacing.0;
                MeasurementValue::RoiStats {
                    mean: 0.0,
                    std_dev: 0.0,
                    min: 0.0,
                    max: 0.0,
                    area_mm2: std::f64::consts::PI * rx_mm * ry_mm,
                    pixel_count: 0,
                }
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
                MeasurementValue::RoiStats {
                    mean: 0.0,
                    std_dev: 0.0,
                    min: 0.0,
                    max: 0.0,
                    area_mm2: area,
                    pixel_count: 0,
                }
            }
            MeasurementKind::PixelProbe { .. } => MeasurementValue::PixelValue {
                value: 0.0,
                unit: "HU".to_string(),
            },
        };

        MeasurementResult {
            measurement_id: self.id.clone(),
            value,
        }
    }
}
