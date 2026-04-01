//! Offscreen volume rendering helpers for pacsleaf.

use crate::lut::ColorLut;
use glam::DVec3;
use leaf_core::domain::SeriesUid;
use leaf_core::error::{LeafError, LeafResult};
use leaf_dicom::volume::assemble_volume;
use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc;
use volren_core::{
    BlendMode, Camera, ColorSpace, ColorTransferFunction, DynVolume, OpacityTransferFunction,
    VolumeInfo, VolumeRenderParams,
};
use volren_gpu::{Viewport, VolumeRenderer};

/// CPU-side RGBA image produced by the volume preview renderer.
#[derive(Clone)]
pub struct VolumePreviewImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Prepared volume data cached for repeated preview renders.
#[derive(Clone)]
pub struct PreparedVolume {
    volume: DynVolume,
}

impl PreparedVolume {
    /// Assemble a DICOM series into a reusable volume.
    pub fn from_series(file_paths: &[String], series_uid: &SeriesUid) -> LeafResult<Self> {
        Ok(Self {
            volume: assemble_volume(file_paths, series_uid)?.into(),
        })
    }

    fn dyn_volume(&self) -> &DynVolume {
        &self.volume
    }
}

/// Pacsleaf-facing volume blend modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VolumeBlendMode {
    #[default]
    Composite,
    MaximumIntensity,
    MinimumIntensity,
    AverageIntensity,
}

impl VolumeBlendMode {
    fn into_volren(self) -> BlendMode {
        match self {
            Self::Composite => BlendMode::Composite,
            Self::MaximumIntensity => BlendMode::MaximumIntensity,
            Self::MinimumIntensity => BlendMode::MinimumIntensity,
            Self::AverageIntensity => BlendMode::AverageIntensity,
        }
    }
}

/// Mutable camera/render state for an interactive volume preview.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeViewState {
    pub azimuth_degrees: f64,
    pub elevation_degrees: f64,
    pub pan_x: f64,
    pub pan_y: f64,
    pub zoom: f64,
    pub blend_mode: VolumeBlendMode,
}

impl Default for VolumeViewState {
    fn default() -> Self {
        Self {
            azimuth_degrees: 0.0,
            elevation_degrees: 0.0,
            pan_x: 0.0,
            pan_y: 0.0,
            zoom: 1.0,
            blend_mode: VolumeBlendMode::Composite,
        }
    }
}

impl VolumeViewState {
    /// Orbit the camera around the volume center.
    pub fn orbit(&mut self, delta_x: f64, delta_y: f64) {
        self.azimuth_degrees += delta_x;
        self.elevation_degrees = (self.elevation_degrees + delta_y).clamp(-85.0, 85.0);
    }

    /// Pan the camera in the view plane.
    pub fn pan(&mut self, delta_x: f64, delta_y: f64) {
        self.pan_x += delta_x;
        self.pan_y += delta_y;
    }

    /// Multiply the zoom factor, keeping it in a safe range.
    pub fn zoom_by(&mut self, factor: f64) {
        self.zoom = (self.zoom * factor).clamp(0.25, 8.0);
    }

    /// Reset the volume view back to the default camera.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Headless offscreen volume renderer for local series previews.
pub struct VolumePreviewRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: VolumeRenderer,
}

impl VolumePreviewRenderer {
    /// Create a new GPU-backed preview renderer.
    pub fn new() -> LeafResult<Self> {
        let (device, queue) = create_device()?;
        let renderer = with_volren_panic_boundary("initializing the renderer", || {
            Ok(VolumeRenderer::new(
                &device,
                &queue,
                wgpu::TextureFormat::Rgba8Unorm,
            ))
        })?;
        Ok(Self {
            device,
            queue,
            renderer,
        })
    }

    /// Assemble a DICOM series into a volume and render a DVR preview.
    pub fn render_series_preview(
        &mut self,
        file_paths: &[String],
        series_uid: &SeriesUid,
        width: u32,
        height: u32,
    ) -> LeafResult<VolumePreviewImage> {
        let prepared = PreparedVolume::from_series(file_paths, series_uid)?;
        self.render_prepared_preview(&prepared, &VolumeViewState::default(), width, height)
    }

    /// Assemble and return a reusable volume for repeated rerendering.
    pub fn prepare_series_volume(
        &self,
        file_paths: &[String],
        series_uid: &SeriesUid,
    ) -> LeafResult<PreparedVolume> {
        PreparedVolume::from_series(file_paths, series_uid)
    }

    /// Render a DVR preview for an already assembled volume.
    pub fn render_preview(
        &mut self,
        volume: &DynVolume,
        width: u32,
        height: u32,
    ) -> LeafResult<VolumePreviewImage> {
        self.render_prepared_preview(
            &PreparedVolume {
                volume: volume.clone(),
            },
            &VolumeViewState::default(),
            width,
            height,
        )
    }

    /// Render a preview for a previously prepared volume and camera state.
    pub fn render_prepared_preview(
        &mut self,
        prepared: &PreparedVolume,
        view_state: &VolumeViewState,
        width: u32,
        height: u32,
    ) -> LeafResult<VolumePreviewImage> {
        if width == 0 || height == 0 {
            return Err(LeafError::InvalidArgument(
                "Volume preview dimensions must be non-zero".into(),
            ));
        }

        let volume = prepared.dyn_volume();
        let render_params = render_params_for_state(volume, view_state);
        let camera = camera_for_state(volume, view_state);
        with_volren_panic_boundary("rendering the volume preview", || {
            self.renderer.set_volume(volume, true);
            self.renderer
                .set_render_params(&render_params)
                .map_err(|error| LeafError::Render(error.to_string()))?;

            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pacsleaf_volume_preview_target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.renderer.output_format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pacsleaf_volume_preview_encoder"),
                });
            clear_render_target(&mut encoder, &view);
            self.renderer
                .render_volume(
                    &mut encoder,
                    &view,
                    &camera,
                    &render_params,
                    Viewport::full(width, height),
                )
                .map_err(|error| LeafError::Render(error.to_string()))?;
            self.queue.submit(std::iter::once(encoder.finish()));

            Ok(VolumePreviewImage {
                width,
                height,
                rgba: read_texture(&self.device, &self.queue, &texture, width, height)?,
            })
        })
    }
}

fn create_device() -> LeafResult<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| LeafError::Render("No compatible GPU adapter available".into()))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        .map_err(|error| LeafError::Render(format!("Failed to create GPU device: {error}")))
}

fn with_volren_panic_boundary<T>(
    context: &str,
    operation: impl FnOnce() -> LeafResult<T>,
) -> LeafResult<T> {
    match panic::catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => Err(LeafError::Render(describe_volren_panic(context, &payload))),
    }
}

fn describe_volren_panic(context: &str, payload: &Box<dyn Any + Send>) -> String {
    let message = panic_payload_message(payload.as_ref());
    if message.contains("volren_reslice_shader")
        || message.contains("reslice.wgsl")
        || message.contains("expected `;`, found \"num_samples\"")
    {
        return format!(
            "volren-rs crashed while {context}. Upstream issue: `volren-rs/crates/volren-gpu/src/shaders/reslice.wgsl:64` contains WGSL that `wgpu 24` rejects (`let offset = if num_samples == 1u {{ ... }}`)."
        );
    }

    format!("volren-rs crashed while {context}: {message}")
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn clear_render_target(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("pacsleaf_volume_preview_clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
}

fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> LeafResult<Vec<u8>> {
    let unpadded_bytes_per_row = width * 4;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
    let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pacsleaf_volume_preview_readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pacsleaf_volume_preview_copy"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let (sender, receiver) = mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    let _ = device.poll(wgpu::MaintainBase::Wait);
    receiver
        .recv()
        .map_err(|error| LeafError::Render(format!("Failed to receive GPU readback: {error}")))?
        .map_err(|error| LeafError::Render(format!("Failed to map GPU readback: {error}")))?;

    let mapped = buffer.slice(..).get_mapped_range();
    let mut pixels = vec![0u8; (unpadded_bytes_per_row * height) as usize];
    for row in 0..height as usize {
        let src_offset = row * padded_bytes_per_row as usize;
        let dst_offset = row * unpadded_bytes_per_row as usize;
        pixels[dst_offset..dst_offset + unpadded_bytes_per_row as usize]
            .copy_from_slice(&mapped[src_offset..src_offset + unpadded_bytes_per_row as usize]);
    }
    drop(mapped);
    buffer.unmap();
    Ok(pixels)
}

fn default_camera(volume: &DynVolume) -> Camera {
    let bounds = volume.world_bounds();
    let center = bounds.center();
    let diagonal = bounds.diagonal().max(1.0);
    Camera::new_perspective(
        center + DVec3::new(diagonal * 0.9, -diagonal * 1.1, diagonal * 0.9),
        center,
        32.0,
    )
    .with_clip_range((diagonal * 0.01).max(0.1), diagonal * 6.0)
}

fn camera_for_state(volume: &DynVolume, view_state: &VolumeViewState) -> Camera {
    let bounds = volume.world_bounds();
    let diagonal = bounds.diagonal().max(1.0);
    let mut camera = default_camera(volume);
    camera.azimuth(view_state.azimuth_degrees);
    camera.elevation(view_state.elevation_degrees);
    camera.pan_view(
        view_state.pan_x * diagonal * 0.0015,
        -view_state.pan_y * diagonal * 0.0015,
    );
    camera.zoom(1.0 / view_state.zoom.clamp(0.25, 8.0));
    camera
}

fn default_render_params(volume: &DynVolume) -> VolumeRenderParams {
    let (scalar_min, scalar_max) = volume.scalar_range();
    VolumeRenderParams::builder()
        .blend_mode(BlendMode::Composite)
        .color_tf(bone_color_transfer_function(scalar_min, scalar_max))
        .opacity_tf(preview_opacity_transfer_function(scalar_min, scalar_max))
        .step_size_factor(0.35)
        .build()
}

fn render_params_for_state(volume: &DynVolume, view_state: &VolumeViewState) -> VolumeRenderParams {
    let (scalar_min, scalar_max) = volume.scalar_range();
    match view_state.blend_mode {
        VolumeBlendMode::Composite => default_render_params(volume),
        VolumeBlendMode::MaximumIntensity
        | VolumeBlendMode::MinimumIntensity
        | VolumeBlendMode::AverageIntensity => VolumeRenderParams::builder()
            .blend_mode(view_state.blend_mode.into_volren())
            .step_size_factor(0.35)
            .color_tf(ColorTransferFunction::greyscale(scalar_min, scalar_max))
            .opacity_tf(OpacityTransferFunction::linear_ramp(scalar_min, scalar_max))
            .build(),
    }
}

fn bone_color_transfer_function(scalar_min: f64, scalar_max: f64) -> ColorTransferFunction {
    let lut = ColorLut::bone();
    let mut color_tf = ColorTransferFunction::new(ColorSpace::Rgb);
    let span = (scalar_max - scalar_min).max(1.0);
    for (index, entry) in lut.table.iter().enumerate() {
        let t = index as f64 / (lut.table.len().saturating_sub(1)) as f64;
        let scalar = scalar_min + span * t;
        color_tf.add_point(
            scalar,
            [
                f64::from(entry[0]) / 255.0,
                f64::from(entry[1]) / 255.0,
                f64::from(entry[2]) / 255.0,
            ],
        );
    }
    color_tf
}

fn preview_opacity_transfer_function(scalar_min: f64, scalar_max: f64) -> OpacityTransferFunction {
    let mut opacity_tf = OpacityTransferFunction::new();
    if scalar_min < -500.0 && scalar_max > 500.0 {
        opacity_tf.add_point(scalar_min, 0.0);
        opacity_tf.add_point(-400.0, 0.0);
        opacity_tf.add_point(100.0, 0.04);
        opacity_tf.add_point(300.0, 0.16);
        opacity_tf.add_point(900.0, 0.55);
        opacity_tf.add_point(scalar_max, 0.88);
        return opacity_tf;
    }

    let span = (scalar_max - scalar_min).max(1.0);
    opacity_tf.add_point(scalar_min, 0.0);
    opacity_tf.add_point(scalar_min + span * 0.18, 0.0);
    opacity_tf.add_point(scalar_min + span * 0.45, 0.05);
    opacity_tf.add_point(scalar_min + span * 0.72, 0.28);
    opacity_tf.add_point(scalar_max, 0.82);
    opacity_tf
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat3, UVec3};
    use volren_core::Volume;

    fn demo_volume() -> DynVolume {
        Volume::from_data(
            vec![0i16, 500, 1000, 1500, 2000, 2500, 3000, 3500],
            UVec3::new(2, 2, 2),
            DVec3::ONE,
            DVec3::ZERO,
            DMat3::IDENTITY,
            1,
        )
        .expect("valid demo volume")
        .into()
    }

    #[test]
    fn preview_camera_targets_volume_center() {
        let volume = demo_volume();
        let camera = default_camera(&volume);
        let center = volume.world_bounds().center();
        assert!((camera.focal_point() - center).length() < 1e-6);
        assert!(camera.distance() > volume.world_bounds().diagonal());
    }

    #[test]
    fn preview_params_use_composite_rendering() {
        let volume = demo_volume();
        let params = default_render_params(&volume);
        assert!(matches!(params.blend_mode, BlendMode::Composite));
        assert!(params.step_size_factor > 0.0);
        assert!(params.opacity_tf.len() >= 4);
        assert!(params.color_tf.len() >= 2);
    }

    #[test]
    fn volume_view_state_clamps_zoom_and_elevation() {
        let mut state = VolumeViewState::default();
        state.orbit(10.0, 200.0);
        state.zoom_by(100.0);
        assert_eq!(state.azimuth_degrees, 10.0);
        assert_eq!(state.elevation_degrees, 85.0);
        assert_eq!(state.zoom, 8.0);
    }

    #[test]
    fn mip_preview_uses_mip_blend_mode() {
        let volume = demo_volume();
        let params = render_params_for_state(
            &volume,
            &VolumeViewState {
                blend_mode: VolumeBlendMode::MaximumIntensity,
                ..VolumeViewState::default()
            },
        );
        assert!(matches!(params.blend_mode, BlendMode::MaximumIntensity));
    }

    #[test]
    fn upstream_shader_panic_is_reported_concisely() {
        let error = with_volren_panic_boundary("initializing the renderer", || -> LeafResult<()> {
            panic!("wgpu error: Validation Error\nShader 'volren_reslice_shader' parsing error: expected `;`, found \"num_samples\"");
        })
        .expect_err("should surface panic as error");
        let message = error.to_string();
        assert!(message.contains("volren-rs/crates/volren-gpu/src/shaders/reslice.wgsl:64"));
        assert!(message.contains("wgpu 24"));
    }
}
