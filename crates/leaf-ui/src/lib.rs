//! Slint UI layer for pacsleaf.
//!
//! Compiles and re-exports the Slint-generated types, and provides
//! the glue code connecting UI callbacks to application logic.

slint::include_modules!();

pub fn image_from_rgba8(
    width: u32,
    height: u32,
    rgba: Vec<u8>,
) -> Result<slint::Image, slint::PlatformError> {
    let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(width, height);
    buffer.make_mut_bytes().copy_from_slice(&rgba);
    Ok(slint::Image::from_rgba8(buffer))
}
