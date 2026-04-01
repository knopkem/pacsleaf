//! Color Lookup Table (LUT) definitions for medical imaging.

/// A color lookup table mapping scalar values to RGBA colors.
pub struct ColorLut {
    pub name: String,
    /// 256 entries, each [R, G, B, A].
    pub table: [[u8; 4]; 256],
}

impl ColorLut {
    /// Standard grayscale LUT.
    pub fn grayscale() -> Self {
        let mut table = [[0u8; 4]; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let v = i as u8;
            *entry = [v, v, v, 255];
        }
        Self {
            name: "grayscale".to_string(),
            table,
        }
    }

    /// Inverted grayscale.
    pub fn grayscale_inverted() -> Self {
        let mut table = [[0u8; 4]; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let v = 255 - i as u8;
            *entry = [v, v, v, 255];
        }
        Self {
            name: "grayscale_inverted".to_string(),
            table,
        }
    }

    /// Hot iron color map (commonly used in medical imaging).
    pub fn hot_iron() -> Self {
        let mut table = [[0u8; 4]; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let t = i as f32 / 255.0;
            let r = (t * 3.0).min(1.0);
            let g = ((t - 0.333) * 3.0).max(0.0).min(1.0);
            let b = ((t - 0.666) * 3.0).max(0.0).min(1.0);
            *entry = [
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (b * 255.0) as u8,
                255,
            ];
        }
        Self {
            name: "hot_iron".to_string(),
            table,
        }
    }

    /// Bone color map.
    pub fn bone() -> Self {
        let mut table = [[0u8; 4]; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let t = i as f32 / 255.0;
            let r = if t < 0.75 {
                t * 8.0 / 9.0
            } else {
                (2.0 * t - 1.0) / 3.0 + 2.0 / 3.0
            };
            let g = if t < 0.375 {
                t * 8.0 / 9.0
            } else if t < 0.75 {
                (2.0 * t - 1.0) / 3.0 + 1.0 / 3.0
            } else {
                (t + 1.0) / 2.0
            };
            let b = if t < 0.375 {
                (t + 1.0 / 9.0) * 9.0 / 8.0 - 1.0 / 8.0
            } else {
                t
            };
            *entry = [
                (r.clamp(0.0, 1.0) * 255.0) as u8,
                (g.clamp(0.0, 1.0) * 255.0) as u8,
                (b.clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ];
        }
        Self {
            name: "bone".to_string(),
            table,
        }
    }

    /// Get all built-in LUTs.
    pub fn all_builtin() -> Vec<Self> {
        vec![
            Self::grayscale(),
            Self::grayscale_inverted(),
            Self::hot_iron(),
            Self::bone(),
        ]
    }
}
