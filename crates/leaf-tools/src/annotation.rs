//! Text annotation support.

use glam::DVec2;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A text annotation placed on an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub id: String,
    pub text: String,
    pub position: DVec2,
    pub series_uid: String,
    pub slice_index: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Annotation {
    pub fn new(text: &str, position: DVec2, series_uid: &str, slice_index: usize) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            text: text.to_string(),
            position,
            series_uid: series_uid.to_string(),
            slice_index,
            created_at: chrono::Utc::now(),
        }
    }
}
