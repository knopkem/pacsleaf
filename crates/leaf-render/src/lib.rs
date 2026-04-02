//! Rendering engine for pacsleaf.
//!
//! Integrates volren-rs for volume rendering and provides
//! 2D viewport management with overlays.

pub mod lut;
pub mod viewport;
pub mod volume;

pub use volren_core::SlicePlane;
pub use volume::{
    PreparedVolume, SlicePreviewMode, SlicePreviewState, SliceProjectionMode, VolumeBlendMode,
    VolumePreviewImage, VolumePreviewRenderer, VolumeViewState,
};
