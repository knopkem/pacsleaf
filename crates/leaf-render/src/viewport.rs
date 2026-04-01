//! Viewport state management for 2D and 3D rendering.

use glam::DVec2;
use volren_core::camera::Camera;
use volren_core::window_level::WindowLevel;

/// State of a single viewport in the viewer.
pub struct ViewportState {
    /// Current window/level settings.
    pub window_level: WindowLevel,
    /// Zoom level (1.0 = fit to viewport).
    pub zoom: f64,
    /// Pan offset in viewport coordinates.
    pub pan: DVec2,
    /// Rotation angle in degrees (for 2D views).
    pub rotation: f64,
    /// Whether the image is horizontally flipped.
    pub flip_h: bool,
    /// Whether the image is vertically flipped.
    pub flip_v: bool,
    /// Whether the image is inverted.
    pub invert: bool,
    /// Active color LUT name.
    pub lut_name: String,
    /// Current slice index in the stack.
    pub slice_index: usize,
    /// Total number of slices.
    pub slice_count: usize,
    /// Camera for 3D/volume views.
    pub camera: Camera,
    /// Current viewport dimensions in pixels.
    pub width: u32,
    pub height: u32,
}

impl ViewportState {
    pub fn new() -> Self {
        Self {
            window_level: WindowLevel::new(40.0, 400.0),
            zoom: 1.0,
            pan: DVec2::ZERO,
            rotation: 0.0,
            flip_h: false,
            flip_v: false,
            invert: false,
            lut_name: "grayscale".to_string(),
            slice_index: 0,
            slice_count: 0,
            camera: Camera::default(),
            width: 512,
            height: 512,
        }
    }

    /// Adjust window/level by mouse drag deltas.
    pub fn adjust_window_level(&mut self, dx: f64, dy: f64) {
        self.window_level.adjust_width(dx);
        self.window_level.adjust_center(-dy);
    }

    /// Scroll through slices.
    pub fn scroll(&mut self, delta: i32) {
        let new_idx = self.slice_index as i32 + delta;
        self.slice_index = new_idx.clamp(0, self.slice_count.saturating_sub(1) as i32) as usize;
    }

    /// Reset all transforms to defaults.
    pub fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan = DVec2::ZERO;
        self.rotation = 0.0;
        self.flip_h = false;
        self.flip_v = false;
        self.invert = false;
    }
}

impl Default for ViewportState {
    fn default() -> Self {
        Self::new()
    }
}
