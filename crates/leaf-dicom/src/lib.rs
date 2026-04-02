//! DICOM bridge — wraps dicom-toolkit-rs for pacsleaf.
//!
//! Provides metadata extraction, pixel data decoding, and volume assembly.

pub mod metadata;
pub mod overlay;
pub mod pixel;
pub mod volume;
