//! Offscreen volume rendering helpers for pacsleaf.

use glam::{DQuat, DVec3};
use leaf_core::domain::SeriesUid;
use leaf_core::error::{LeafError, LeafResult};
use leaf_dicom::volume::assemble_volume;
use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc;
use tracing::info;
use volren_core::{
    Aabb, BlendMode, Camera, ColorSpace, ColorTransferFunction, DynVolume, OpacityTransferFunction,
    Projection, ShadingParams, SlicePlane, ThickSlabMode, ThickSlabParams, VolumeInfo,
    VolumeRenderParams, WindowLevel,
};
use volren_gpu::{Viewport, VolumeRenderer};

/// CPU-side RGBA image produced by the volume preview renderer.
#[derive(Clone)]
pub struct VolumePreviewImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

struct CachedPreviewTarget {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
}

struct CachedReadbackBuffer {
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    buffer: wgpu::Buffer,
}

/// Prepared volume data cached for repeated preview renders.
#[derive(Clone)]
pub struct PreparedVolume {
    volume: DynVolume,
    cache_key: String,
}

impl PreparedVolume {
    /// Assemble a DICOM series into a reusable volume.
    pub fn from_series(file_paths: &[String], series_uid: &SeriesUid) -> LeafResult<Self> {
        let volume: DynVolume = assemble_volume(file_paths, series_uid)?.into();

        // Log volume geometry for camera debugging.
        let bounds = volume.world_bounds();
        let dir = volume.direction();
        info!(
            "DVR volume prepared: dims={:?} spacing={:?} origin={:?} \
             direction=[col0={:?}, col1={:?}, col2={:?}] \
             bounds=[{:?} .. {:?}] center={:?} diagonal={:.1} \
             scalar_range={:?}",
            volume.dimensions(),
            volume.spacing(),
            volume.origin(),
            dir.col(0),
            dir.col(1),
            dir.col(2),
            bounds.min,
            bounds.max,
            bounds.center(),
            bounds.diagonal(),
            volume.scalar_range(),
        );

        Ok(Self {
            volume,
            cache_key: series_uid.0.clone(),
        })
    }

    fn dyn_volume(&self) -> &DynVolume {
        &self.volume
    }

    fn cache_key(&self) -> &str {
        &self.cache_key
    }

    /// Scalar range of the prepared volume.
    pub fn scalar_range(&self) -> (f64, f64) {
        self.volume.scalar_range()
    }

    /// World-space bounds of the prepared volume.
    pub fn world_bounds(&self) -> Aabb {
        self.volume.world_bounds()
    }

    /// Suggested scroll step for axis-aligned MPR slicing.
    pub fn slice_scroll_step(&self, mode: SlicePreviewMode) -> f64 {
        let spacing = self.volume.spacing();
        match mode {
            SlicePreviewMode::Axial => spacing.z.abs(),
            SlicePreviewMode::Coronal => spacing.y.abs(),
            SlicePreviewMode::Sagittal => spacing.x.abs(),
        }
        .max(0.5)
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

/// Axis-aligned single-viewport MPR modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlicePreviewMode {
    #[default]
    Axial,
    Coronal,
    Sagittal,
}

/// Projection style for a single-slice or slab MPR preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceProjectionMode {
    #[default]
    Thin,
    MaximumIntensity,
    MinimumIntensity,
    AverageIntensity,
}

/// Mutable state for a single MPR slice preview.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlicePreviewState {
    pub mode: SlicePreviewMode,
    pub offset: f64,
    pub orientation: DQuat,
    pub projection_mode: SliceProjectionMode,
    pub slab_half_thickness: f64,
    pub crosshair_world: Option<DVec3>,
    pub transfer_center_hu: Option<f64>,
    pub transfer_width_hu: Option<f64>,
    slab_settings_by_mode: [SliceSlabSettings; 3],
}

impl Default for SlicePreviewState {
    fn default() -> Self {
        let slab_settings = [SliceSlabSettings::default(); 3];
        Self {
            mode: SlicePreviewMode::Axial,
            offset: 0.0,
            orientation: DQuat::IDENTITY,
            projection_mode: slab_settings[0].projection_mode,
            slab_half_thickness: slab_settings[0].slab_half_thickness,
            crosshair_world: None,
            transfer_center_hu: None,
            transfer_width_hu: None,
            slab_settings_by_mode: slab_settings,
        }
    }
}

impl SlicePreviewState {
    /// Ensure the reslice preview has a modality-appropriate window.
    pub fn ensure_transfer_window(&mut self, scalar_min: f64, scalar_max: f64) {
        let (center, width) = resolved_slice_transfer_window(*self, scalar_min, scalar_max);
        self.transfer_center_hu.get_or_insert(center);
        self.transfer_width_hu.get_or_insert(width);
    }

    /// Read the current slice window, falling back to sensible defaults.
    pub fn transfer_window(&self, scalar_min: f64, scalar_max: f64) -> (f64, f64) {
        resolved_slice_transfer_window(*self, scalar_min, scalar_max)
    }

    /// Update the slice transfer window while keeping it in a safe range.
    pub fn set_transfer_window(
        &mut self,
        center: f64,
        width: f64,
        scalar_min: f64,
        scalar_max: f64,
    ) {
        let (center, width) = clamp_transfer_window(center, width, scalar_min, scalar_max);
        self.transfer_center_hu = Some(center);
        self.transfer_width_hu = Some(width);
    }

    /// Reset the preview back to the centred default slice for the current orientation.
    pub fn reset(&mut self) {
        self.offset = 0.0;
        self.orientation = DQuat::IDENTITY;
        self.projection_mode = SliceProjectionMode::Thin;
        self.slab_half_thickness = 0.0;
        self.crosshair_world = None;
        self.transfer_center_hu = None;
        self.transfer_width_hu = None;
        self.slab_settings_by_mode = [SliceSlabSettings::default(); 3];
    }

    /// Switch to a new orthogonal slice mode.
    pub fn set_mode(&mut self, mode: SlicePreviewMode) {
        self.persist_current_slab_settings();
        self.mode = mode;
        self.restore_current_slab_settings();
    }

    /// Resolve the current slice plane against a prepared volume bound.
    pub fn slice_plane(&self, bounds: Aabb) -> SlicePlane {
        slice_plane_for_state(bounds, *self)
    }

    /// The active crosshair world point, defaulting to the volume center.
    pub fn crosshair_world(&self, bounds: Aabb) -> DVec3 {
        self.crosshair_world.unwrap_or(bounds.center())
    }

    /// Update the shared crosshair point in world space.
    pub fn set_crosshair_world(&mut self, world: DVec3) {
        self.crosshair_world = Some(world);
    }

    /// Move the current slice so it passes through the given world-space point.
    pub fn center_on_world(&mut self, world: DVec3, bounds: Aabb) {
        let center = bounds.center();
        let normal = self.slice_plane(bounds).normal();
        let unclamped_offset = (world - center).dot(normal);
        self.offset = unclamped_offset;
        self.clamp_offset(bounds);
        self.crosshair_world = Some(world + normal * (self.offset - unclamped_offset));
    }

    /// Move the current slice so it passes through the active crosshair point.
    pub fn center_on_crosshair(&mut self, bounds: Aabb) {
        let world = self.crosshair_world(bounds);
        self.center_on_world(world, bounds);
    }

    /// Advance to the next slab projection mode.
    pub fn cycle_projection_mode(&mut self, default_half_thickness: f64) {
        self.projection_mode = match self.projection_mode {
            SliceProjectionMode::Thin => SliceProjectionMode::MaximumIntensity,
            SliceProjectionMode::MaximumIntensity => SliceProjectionMode::MinimumIntensity,
            SliceProjectionMode::MinimumIntensity => SliceProjectionMode::AverageIntensity,
            SliceProjectionMode::AverageIntensity => SliceProjectionMode::Thin,
        };
        self.slab_half_thickness = if matches!(self.projection_mode, SliceProjectionMode::Thin) {
            0.0
        } else {
            default_half_thickness.max(0.5)
        };
        self.persist_current_slab_settings();
    }

    /// Update the current mode's slab thickness from a drag handle.
    pub fn set_slab_half_thickness_from_drag(
        &mut self,
        half_thickness: f64,
        min_active_half_thickness: f64,
        fallback_mode: SliceProjectionMode,
    ) {
        if half_thickness <= min_active_half_thickness {
            self.projection_mode = SliceProjectionMode::Thin;
            self.slab_half_thickness = 0.0;
        } else {
            if matches!(self.projection_mode, SliceProjectionMode::Thin) {
                self.projection_mode = fallback_mode;
            }
            self.slab_half_thickness = half_thickness.max(0.5);
        }
        self.persist_current_slab_settings();
    }

    /// Resolve thick-slab parameters for the current projection mode.
    pub fn thick_slab(self) -> Option<ThickSlabParams> {
        let mode = match self.projection_mode {
            SliceProjectionMode::Thin => return None,
            SliceProjectionMode::MaximumIntensity => ThickSlabMode::Mip,
            SliceProjectionMode::MinimumIntensity => ThickSlabMode::MinIp,
            SliceProjectionMode::AverageIntensity => ThickSlabMode::Mean,
        };
        Some(ThickSlabParams {
            half_thickness: self.slab_half_thickness.max(0.5),
            mode,
            num_samples: 16,
        })
    }

    /// Clamp the current offset to the volume bounds for the active axis.
    pub fn clamp_offset(&mut self, bounds: Aabb) {
        let (min_offset, max_offset) =
            slice_offset_range(bounds, self.slice_plane(bounds).normal());
        self.offset = self.offset.clamp(min_offset, max_offset);
    }

    /// Scroll the current slice by `delta` world units and keep it within bounds.
    pub fn scroll_by(&mut self, delta: f64, bounds: Aabb) {
        let world = self.crosshair_world(bounds) + self.slice_plane(bounds).normal() * delta;
        self.center_on_world(world, bounds);
    }

    /// Rotate the shared MPR cursor around the current plane normal.
    pub fn rotate_about_normal(&mut self, angle_rad: f64, bounds: Aabb) {
        let axis = self.slice_plane(bounds).normal();
        let rotation = DQuat::from_axis_angle(axis.normalize_or(DVec3::Z), angle_rad);
        self.orientation = (rotation * self.orientation).normalize();
        self.center_on_crosshair(bounds);
    }

    fn persist_current_slab_settings(&mut self) {
        self.slab_settings_by_mode[mode_index(self.mode)] = SliceSlabSettings {
            projection_mode: self.projection_mode,
            slab_half_thickness: self.slab_half_thickness,
        };
    }

    fn restore_current_slab_settings(&mut self) {
        let settings = self.slab_settings_by_mode[mode_index(self.mode)];
        self.projection_mode = settings.projection_mode;
        self.slab_half_thickness = settings.slab_half_thickness;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SliceSlabSettings {
    projection_mode: SliceProjectionMode,
    slab_half_thickness: f64,
}

impl Default for SliceSlabSettings {
    fn default() -> Self {
        Self {
            projection_mode: SliceProjectionMode::Thin,
            slab_half_thickness: 0.0,
        }
    }
}

fn mode_index(mode: SlicePreviewMode) -> usize {
    match mode {
        SlicePreviewMode::Axial => 0,
        SlicePreviewMode::Coronal => 1,
        SlicePreviewMode::Sagittal => 2,
    }
}

/// Mutable camera/render state for an interactive volume preview.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeViewState {
    /// Accumulated rotation quaternion from the default AP view.
    pub orientation: DQuat,
    pub pan_x: f64,
    pub pan_y: f64,
    pub zoom: f64,
    pub blend_mode: VolumeBlendMode,
    pub transfer_center_hu: Option<f64>,
    pub transfer_width_hu: Option<f64>,
}

impl Default for VolumeViewState {
    fn default() -> Self {
        Self {
            orientation: DQuat::IDENTITY,
            pan_x: 0.0,
            pan_y: 0.0,
            zoom: 1.0,
            blend_mode: VolumeBlendMode::Composite,
            transfer_center_hu: None,
            transfer_width_hu: None,
        }
    }
}

impl VolumeViewState {
    /// Orbit the camera around the volume center (unlimited, quaternion-based).
    ///
    /// `delta_x` rotates around the world Z axis (turntable).
    /// `delta_y` rotates around the camera's local right axis (tumble).
    pub fn orbit(&mut self, delta_x: f64, delta_y: f64) {
        let yaw = DQuat::from_axis_angle(DVec3::Z, -delta_x.to_radians());
        let local_right = self.orientation * DVec3::X;
        let pitch = DQuat::from_axis_angle(local_right, -delta_y.to_radians());
        self.orientation = (pitch * yaw * self.orientation).normalize();
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

    /// Ensure the transfer window has a modality-appropriate default.
    pub fn ensure_transfer_window(&mut self, scalar_min: f64, scalar_max: f64) {
        let (center, width) = resolved_transfer_window(*self, scalar_min, scalar_max);
        self.transfer_center_hu.get_or_insert(center);
        self.transfer_width_hu.get_or_insert(width);
    }

    /// Read the current transfer window, falling back to sensible defaults.
    pub fn transfer_window(&self, scalar_min: f64, scalar_max: f64) -> (f64, f64) {
        resolved_transfer_window(*self, scalar_min, scalar_max)
    }

    /// Update the transfer window while keeping it in a safe range for the volume.
    pub fn set_transfer_window(
        &mut self,
        center: f64,
        width: f64,
        scalar_min: f64,
        scalar_max: f64,
    ) {
        let (center, width) = clamp_transfer_window(center, width, scalar_min, scalar_max);
        self.transfer_center_hu = Some(center);
        self.transfer_width_hu = Some(width);
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
    uploaded_volume_key: Option<String>,
    preview_target: Option<CachedPreviewTarget>,
    readback_buffer: Option<CachedReadbackBuffer>,
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
            uploaded_volume_key: None,
            preview_target: None,
            readback_buffer: None,
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
        self.render_prepared_preview(&prepared, &VolumeViewState::default(), width, height, false)
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
                cache_key: format!("__ad_hoc_preview__:{:p}", volume),
            },
            &VolumeViewState::default(),
            width,
            height,
            false,
        )
    }

    /// Render a preview for a previously prepared volume and camera state.
    pub fn render_prepared_preview(
        &mut self,
        prepared: &PreparedVolume,
        view_state: &VolumeViewState,
        width: u32,
        height: u32,
        interactive: bool,
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
            self.ensure_prepared_volume_uploaded(prepared);
            self.renderer
                .set_render_params(&render_params)
                .map_err(|error| LeafError::Render(error.to_string()))?;
            self.ensure_preview_target(width, height);
            self.ensure_readback_buffer(width, height);

            let texture = &self
                .preview_target
                .as_ref()
                .ok_or_else(|| LeafError::Render("Preview target unavailable".into()))?
                .texture;
            let readback = self
                .readback_buffer
                .as_ref()
                .ok_or_else(|| LeafError::Render("Preview readback buffer unavailable".into()))?;
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pacsleaf_volume_preview_encoder"),
                });
            clear_render_target(&mut encoder, &view);
            let viewport = Viewport::full(width, height);
            if interactive {
                self.renderer
                    .render_volume_interactive(
                        &mut encoder,
                        &view,
                        &camera,
                        &render_params,
                        viewport,
                        interactive_downsample_factor(width, height),
                    )
                    .map_err(|error| LeafError::Render(error.to_string()))?;
            } else {
                self.renderer
                    .render_volume(&mut encoder, &view, &camera, &render_params, viewport)
                    .map_err(|error| LeafError::Render(error.to_string()))?;
            }
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(readback.padded_bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));

            Ok(VolumePreviewImage {
                width,
                height,
                rgba: read_buffer(
                    &self.device,
                    &readback.buffer,
                    width,
                    height,
                    readback.padded_bytes_per_row,
                )?,
            })
        })
    }

    /// Render a single orthogonal MPR slice for an already prepared volume.
    pub fn render_prepared_slice_preview(
        &mut self,
        prepared: &PreparedVolume,
        view_state: &SlicePreviewState,
        width: u32,
        height: u32,
        show_crosshair: bool,
    ) -> LeafResult<VolumePreviewImage> {
        if width == 0 || height == 0 {
            return Err(LeafError::InvalidArgument(
                "Slice preview dimensions must be non-zero".into(),
            ));
        }

        let bounds = prepared.world_bounds();
        let slice_plane = view_state.slice_plane(bounds);
        let (scalar_min, scalar_max) = prepared.scalar_range();
        let (center, width_hu) = view_state.transfer_window(scalar_min, scalar_max);
        let window_level = WindowLevel::new(center, width_hu.max(1.0));
        let thick_slab = view_state.thick_slab();
        let crosshair_world = view_state.crosshair_world(bounds);

        with_volren_panic_boundary("rendering the MPR slice preview", || {
            self.ensure_prepared_volume_uploaded(prepared);
            self.ensure_preview_target(width, height);
            self.ensure_readback_buffer(width, height);

            let texture = &self
                .preview_target
                .as_ref()
                .ok_or_else(|| LeafError::Render("Preview target unavailable".into()))?
                .texture;
            let readback = self
                .readback_buffer
                .as_ref()
                .ok_or_else(|| LeafError::Render("Preview readback buffer unavailable".into()))?;
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pacsleaf_slice_preview_encoder"),
                });
            clear_render_target(&mut encoder, &view);
            self.renderer
                .render_slice(
                    &mut encoder,
                    &view,
                    &slice_plane,
                    &window_level,
                    Viewport::full(width, height),
                    thick_slab.as_ref(),
                )
                .map_err(|error| LeafError::Render(error.to_string()))?;
            let (crosshair_uv, _) = slice_plane.world_to_point(crosshair_world);
            if show_crosshair
                && (0.0..=1.0).contains(&crosshair_uv.x)
                && (0.0..=1.0).contains(&crosshair_uv.y)
            {
                self.renderer
                    .render_crosshair(
                        &mut encoder,
                        &view,
                        Viewport::full(width, height),
                        &volren_gpu::CrosshairParams {
                            position: [crosshair_uv.x as f32, crosshair_uv.y as f32],
                            horizontal_color: [1.0, 0.72, 0.2, 0.9],
                            vertical_color: [0.25, 0.85, 1.0, 0.9],
                            thickness: 1.5,
                        },
                    )
                    .map_err(|error| LeafError::Render(error.to_string()))?;
            }
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(readback.padded_bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));

            Ok(VolumePreviewImage {
                width,
                height,
                rgba: read_buffer(
                    &self.device,
                    &readback.buffer,
                    width,
                    height,
                    readback.padded_bytes_per_row,
                )?,
            })
        })
    }

    fn ensure_prepared_volume_uploaded(&mut self, prepared: &PreparedVolume) {
        if self.uploaded_volume_key.as_deref() == Some(prepared.cache_key()) {
            return;
        }

        self.renderer.set_volume(prepared.dyn_volume(), true);
        self.uploaded_volume_key = Some(prepared.cache_key().to_string());
    }

    fn ensure_preview_target(&mut self, width: u32, height: u32) {
        let needs_rebuild = self.preview_target.as_ref().map_or(true, |target| {
            target.width != width || target.height != height
        });
        if !needs_rebuild {
            return;
        }

        self.preview_target = Some(CachedPreviewTarget {
            width,
            height,
            texture: self.device.create_texture(&wgpu::TextureDescriptor {
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
            }),
        });
    }

    fn ensure_readback_buffer(&mut self, width: u32, height: u32) {
        let padded_bytes_per_row = (width * 4).div_ceil(256) * 256;
        let needs_rebuild = self.readback_buffer.as_ref().map_or(true, |buffer| {
            buffer.width != width
                || buffer.height != height
                || buffer.padded_bytes_per_row != padded_bytes_per_row
        });
        if !needs_rebuild {
            return;
        }

        self.readback_buffer = Some(CachedReadbackBuffer {
            width,
            height,
            padded_bytes_per_row,
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pacsleaf_volume_preview_readback"),
                size: u64::from(padded_bytes_per_row) * u64::from(height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        });
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

fn read_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) -> LeafResult<Vec<u8>> {
    let unpadded_bytes_per_row = width * 4;

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

/// Build the camera from scratch each frame using KPACS-style Rodrigues rotation.
///
/// This avoids volren's built-in `orbit()` which hardcodes `DVec3::Y` as the
/// world up-axis — incompatible with DICOM's Z-superior convention.
///
/// Quaternion-based camera positioning for the volume preview.
///
/// Default view: standard radiological AP — camera at the anterior side of
/// the patient looking posteriorly, with superior (head) pointing up.
/// The orientation quaternion rotates from this default without any gimbal
/// lock, allowing unlimited rotation.
fn camera_for_state(volume: &DynVolume, view_state: &VolumeViewState) -> Camera {
    let bounds = volume.world_bounds();
    let center = bounds.center();
    let diagonal = bounds.diagonal().max(1.0);

    // Default AP axes in DICOM LPS: forward = +Y (posterior), up = -Z.
    // The negated Z compensates for the wgpu screen-space Y convention so
    // that the patient's head appears at the top of the viewport.
    let default_forward = DVec3::Y;
    let default_up = DVec3::NEG_Z;

    // Apply the accumulated quaternion orientation.
    let forward = view_state.orientation * default_forward;
    let up = view_state.orientation * default_up;
    let right = forward.cross(up).normalize_or(DVec3::X);

    // Distance: fit the full 3D diagonal inside the viewport with 15% margin.
    let fov_y_deg = 30.0_f64;
    let half_diag = diagonal * 0.5;
    let fit_distance = half_diag / (fov_y_deg.to_radians() * 0.5).tan();
    let distance = fit_distance * 1.15 / view_state.zoom.clamp(0.25, 8.0);
    let position = center - forward * distance;

    // Pan in the view plane, scaled to current distance for consistent
    // screen-space feel at any zoom level.
    let pan_scale = distance * 0.001;
    let pan_offset = right * (-view_state.pan_x * pan_scale) + up * (-view_state.pan_y * pan_scale);

    Camera::new(position + pan_offset, center + pan_offset, up)
        .with_projection(Projection::Perspective { fov_y_deg })
        .with_clip_range(
            (distance - diagonal).max(diagonal * 0.01).max(0.1),
            distance + diagonal * 2.0,
        )
}

fn default_render_params(volume: &DynVolume, view_state: &VolumeViewState) -> VolumeRenderParams {
    let (scalar_min, scalar_max) = volume.scalar_range();
    if looks_ct_like(scalar_min, scalar_max) {
        let (center, width) = view_state.transfer_window(scalar_min, scalar_max);
        return VolumeRenderParams::builder()
            .blend_mode(BlendMode::Composite)
            .color_tf(kpacs_soft_tissue_color_transfer_function(
                scalar_min, scalar_max, center, width,
            ))
            .opacity_tf(kpacs_windowed_soft_tissue_opacity_transfer_function(
                scalar_min, scalar_max, center, width,
            ))
            .shading(shading_for_transfer_window(center))
            .step_size_factor(0.55)
            .build();
    }

    VolumeRenderParams::builder()
        .blend_mode(BlendMode::Composite)
        .color_tf(ColorTransferFunction::greyscale(scalar_min, scalar_max))
        .opacity_tf(generic_opacity_transfer_function(scalar_min, scalar_max))
        .step_size_factor(0.6)
        .build()
}

fn render_params_for_state(volume: &DynVolume, view_state: &VolumeViewState) -> VolumeRenderParams {
    let (scalar_min, scalar_max) = volume.scalar_range();
    match view_state.blend_mode {
        VolumeBlendMode::Composite => default_render_params(volume, view_state),
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

fn looks_ct_like(scalar_min: f64, scalar_max: f64) -> bool {
    scalar_min <= -500.0 && scalar_max >= 1200.0
}

fn interactive_downsample_factor(width: u32, height: u32) -> u32 {
    let _ = (width, height);
    2
}

fn resolved_transfer_window(
    view_state: VolumeViewState,
    scalar_min: f64,
    scalar_max: f64,
) -> (f64, f64) {
    let range = (scalar_max - scalar_min).max(1.0);
    let default_center = if looks_ct_like(scalar_min, scalar_max) {
        90.0
    } else {
        scalar_min + range * 0.5
    };
    let default_width = if looks_ct_like(scalar_min, scalar_max) {
        700.0
    } else {
        range
    };

    clamp_transfer_window(
        view_state.transfer_center_hu.unwrap_or(default_center),
        view_state.transfer_width_hu.unwrap_or(default_width),
        scalar_min,
        scalar_max,
    )
}

fn resolved_slice_transfer_window(
    view_state: SlicePreviewState,
    scalar_min: f64,
    scalar_max: f64,
) -> (f64, f64) {
    let range = (scalar_max - scalar_min).max(1.0);
    clamp_transfer_window(
        view_state
            .transfer_center_hu
            .unwrap_or(scalar_min + range * 0.5),
        view_state.transfer_width_hu.unwrap_or(range),
        scalar_min,
        scalar_max,
    )
}

fn clamp_transfer_window(center: f64, width: f64, scalar_min: f64, scalar_max: f64) -> (f64, f64) {
    let range = (scalar_max - scalar_min).max(1.0);
    (
        center.clamp(scalar_min - range * 0.25, scalar_max + range * 0.25),
        width.clamp(range / 200.0, range * 1.25),
    )
}

fn slice_basis_for_mode(mode: SlicePreviewMode) -> (DVec3, DVec3) {
    match mode {
        SlicePreviewMode::Axial => (DVec3::X, DVec3::Y),
        SlicePreviewMode::Coronal => (DVec3::X, -DVec3::Z),
        SlicePreviewMode::Sagittal => (DVec3::Y, -DVec3::Z),
    }
}

fn slice_preferred_up_for_mode(mode: SlicePreviewMode) -> DVec3 {
    match mode {
        SlicePreviewMode::Axial => DVec3::Y,
        SlicePreviewMode::Coronal | SlicePreviewMode::Sagittal => -DVec3::Z,
    }
}

fn slice_basis_from_normal(mode: SlicePreviewMode, normal: DVec3) -> (DVec3, DVec3) {
    let project_reference = |reference: DVec3| {
        let projected = reference - normal * reference.dot(normal);
        (projected.length_squared() > 1.0e-10).then(|| projected.normalize())
    };

    let up = project_reference(slice_preferred_up_for_mode(mode))
        .or_else(|| {
            [DVec3::X, DVec3::Y, DVec3::Z]
                .into_iter()
                .filter_map(project_reference)
                .next()
        })
        .unwrap_or(DVec3::Y);
    let right = up.cross(normal).normalize_or(DVec3::X);
    let up = normal.cross(right).normalize_or(up);
    (right, up)
}

fn slice_offset_range(bounds: Aabb, normal: DVec3) -> (f64, f64) {
    let center = bounds.center();
    let corners = [
        DVec3::new(bounds.min.x, bounds.min.y, bounds.min.z),
        DVec3::new(bounds.min.x, bounds.min.y, bounds.max.z),
        DVec3::new(bounds.min.x, bounds.max.y, bounds.min.z),
        DVec3::new(bounds.min.x, bounds.max.y, bounds.max.z),
        DVec3::new(bounds.max.x, bounds.min.y, bounds.min.z),
        DVec3::new(bounds.max.x, bounds.min.y, bounds.max.z),
        DVec3::new(bounds.max.x, bounds.max.y, bounds.min.z),
        DVec3::new(bounds.max.x, bounds.max.y, bounds.max.z),
    ];
    let mut min_offset = f64::INFINITY;
    let mut max_offset = f64::NEG_INFINITY;
    for corner in corners {
        let offset = (corner - center).dot(normal);
        min_offset = min_offset.min(offset);
        max_offset = max_offset.max(offset);
    }
    (min_offset, max_offset)
}

fn slice_plane_for_state(bounds: Aabb, view_state: SlicePreviewState) -> SlicePlane {
    let center = bounds.center();
    let size = bounds.size();
    let (base_right, base_up) = slice_basis_for_mode(view_state.mode);
    let default_normal = base_right.cross(base_up).normalize_or(DVec3::Z);
    let normal = (view_state.orientation * default_normal).normalize_or(default_normal);
    let (right, up) = slice_basis_from_normal(view_state.mode, normal);
    let (min_offset, max_offset) = slice_offset_range(bounds, normal);
    let clamped_offset = view_state.offset.clamp(min_offset, max_offset);
    let origin = center + normal * clamped_offset;
    match view_state.mode {
        SlicePreviewMode::Axial => {
            SlicePlane::new(origin, right, up, size.x.max(1.0), size.y.max(1.0))
        }
        SlicePreviewMode::Coronal => {
            SlicePlane::new(origin, right, up, size.x.max(1.0), size.z.max(1.0))
        }
        SlicePreviewMode::Sagittal => {
            SlicePlane::new(origin, right, up, size.y.max(1.0), size.z.max(1.0))
        }
    }
}

fn kpacs_windowed_soft_tissue_opacity_transfer_function(
    scalar_min: f64,
    scalar_max: f64,
    center: f64,
    width: f64,
) -> OpacityTransferFunction {
    const DEFAULT_CENTER: f64 = 90.0;
    const DEFAULT_WIDTH: f64 = 700.0;
    let safe_width = width.max(1.0);
    let mut opacity_tf = OpacityTransferFunction::new();
    let mut previous_value: Option<f64> = None;
    for (value, opacity) in [
        (scalar_min, 0.0),
        (-200.0, 0.0),
        (-100.0, 0.01),
        (0.0, 0.08),
        (40.0, 0.20),
        (80.0, 0.25),
        (200.0, 0.18),
        (300.0, 0.05),
        (500.0, 0.0),
        (scalar_max, 0.0),
    ] {
        let remapped = (center + ((value - DEFAULT_CENTER) / DEFAULT_WIDTH) * safe_width)
            .clamp(scalar_min, scalar_max);
        if previous_value
            .map(|previous| (previous - remapped).abs() <= 1e-6)
            .unwrap_or(false)
        {
            continue;
        }
        opacity_tf.add_point(remapped, opacity);
        previous_value = Some(remapped);
    }
    opacity_tf
}

fn kpacs_soft_tissue_color_transfer_function(
    scalar_min: f64,
    scalar_max: f64,
    center: f64,
    width: f64,
) -> ColorTransferFunction {
    let mut color_tf = ColorTransferFunction::new(ColorSpace::Rgb);
    let samples = 24;
    let span = (scalar_max - scalar_min).max(1.0);
    for index in 0..samples {
        let t = index as f64 / (samples - 1) as f64;
        let scalar = scalar_min + span * t;
        color_tf.add_point(
            scalar,
            sample_kpacs_auto_ct_color(scalar, center, width, 0.72),
        );
    }
    color_tf
}

fn sample_kpacs_auto_ct_color(
    scalar_value: f64,
    focus_center: f64,
    focus_width: f64,
    color_strength: f64,
) -> [f64; 3] {
    let safe_focus_width = focus_width.max(1.0);
    let target_luminance = ((scalar_value - (focus_center - safe_focus_width * 0.5))
        / safe_focus_width)
        .clamp(0.0, 1.0);
    let base_color = interpolate_ct_anchor_color(scalar_value);
    let matched_color = match_luminance(base_color, target_luminance);
    let focus_distance = (scalar_value - focus_center).abs() / safe_focus_width;
    let focus_weight = 1.0 / (1.0 + focus_distance * focus_distance * 6.0);
    let blend = (0.14 + focus_weight * color_strength).clamp(0.0, 0.95);
    let grayscale = RgbColor::new(target_luminance, target_luminance, target_luminance);
    let final_color = match_luminance(grayscale.lerp(matched_color, blend), target_luminance);
    [final_color.r, final_color.g, final_color.b]
}

fn interpolate_ct_anchor_color(scalar_value: f64) -> RgbColor {
    const ANCHORS: &[(f64, RgbColor)] = &[
        (-1000.0, RgbColor::new(0.00, 0.00, 0.00)),
        (-700.0, RgbColor::new(0.63, 0.47, 0.43)),
        (-120.0, RgbColor::new(0.83, 0.74, 0.55)),
        (60.0, RgbColor::new(0.67, 0.17, 0.17)),
        (180.0, RgbColor::new(0.84, 0.30, 0.30)),
        (450.0, RgbColor::new(0.90, 0.88, 0.82)),
        (1200.0, RgbColor::new(1.00, 1.00, 1.00)),
    ];

    if scalar_value <= ANCHORS[0].0 {
        return ANCHORS[0].1;
    }
    if scalar_value >= ANCHORS[ANCHORS.len() - 1].0 {
        return ANCHORS[ANCHORS.len() - 1].1;
    }

    for window in ANCHORS.windows(2) {
        let (start_value, start_color) = window[0];
        let (end_value, end_color) = window[1];
        if scalar_value < start_value || scalar_value > end_value {
            continue;
        }
        let span = (end_value - start_value).max(1e-6);
        let t = (scalar_value - start_value) / span;
        return start_color.lerp(end_color, t);
    }

    ANCHORS[ANCHORS.len() - 1].1
}

fn match_luminance(color: RgbColor, target_luminance: f64) -> RgbColor {
    let luminance = color.luminance().max(1e-6);
    let scale = target_luminance / luminance;
    RgbColor::new(color.r * scale, color.g * scale, color.b * scale).clamp01()
}

fn shading_for_transfer_window(center: f64) -> ShadingParams {
    if center >= 400.0 {
        // Bone / hard tissue: sharp specular highlights.
        ShadingParams {
            ambient: 0.15,
            diffuse: 0.50,
            specular: 1.05,
            specular_power: 54.0,
        }
    } else {
        // Soft tissue: brighter ambient prevents dark regions,
        // moderate specular adds definition (matching mediseen VTK).
        ShadingParams {
            ambient: 0.45,
            diffuse: 0.70,
            specular: 0.60,
            specular_power: 17.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RgbColor {
    r: f64,
    g: f64,
    b: f64,
}

impl RgbColor {
    const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    fn luminance(self) -> f64 {
        self.r * 0.2126 + self.g * 0.7152 + self.b * 0.0722
    }

    fn lerp(self, other: Self, t: f64) -> Self {
        Self::new(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
        )
    }

    fn clamp01(self) -> Self {
        Self::new(
            self.r.clamp(0.0, 1.0),
            self.g.clamp(0.0, 1.0),
            self.b.clamp(0.0, 1.0),
        )
    }
}

fn generic_opacity_transfer_function(scalar_min: f64, scalar_max: f64) -> OpacityTransferFunction {
    let span = (scalar_max - scalar_min).max(1.0);
    let mut opacity_tf = OpacityTransferFunction::new();
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
    use std::time::Instant;
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

    fn synthetic_ct_volume() -> DynVolume {
        let size = UVec3::new(96, 96, 96);
        let center = DVec3::new(
            f64::from(size.x - 1) * 0.5,
            f64::from(size.y - 1) * 0.5,
            f64::from(size.z - 1) * 0.5,
        );
        let mut voxels = Vec::with_capacity((size.x * size.y * size.z) as usize);

        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let point = DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                    let radius = (point - center).length() / 42.0;
                    let value = if radius > 1.0 {
                        -1000
                    } else if radius > 0.88 {
                        40
                    } else if radius > 0.74 {
                        950
                    } else {
                        55
                    };
                    voxels.push(value);
                }
            }
        }

        Volume::from_data(voxels, size, DVec3::ONE, DVec3::ZERO, DMat3::IDENTITY, 1)
            .expect("valid synthetic CT volume")
            .into()
    }

    fn z_gradient_volume() -> DynVolume {
        let size = UVec3::new(16, 16, 16);
        let mut voxels = Vec::with_capacity((size.x * size.y * size.z) as usize);
        for z in 0..size.z {
            for _y in 0..size.y {
                for _x in 0..size.x {
                    voxels.push((z as i16) * 200);
                }
            }
        }
        Volume::from_data(voxels, size, DVec3::ONE, DVec3::ZERO, DMat3::IDENTITY, 1)
            .expect("valid z-gradient volume")
            .into()
    }

    fn average_row_luma(image: &VolumePreviewImage, row: u32) -> f64 {
        let start = (row * image.width * 4) as usize;
        let end = start + (image.width * 4) as usize;
        image.rgba[start..end]
            .chunks_exact(4)
            .map(|px| (f64::from(px[0]) + f64::from(px[1]) + f64::from(px[2])) / 3.0)
            .sum::<f64>()
            / f64::from(image.width)
    }

    #[test]
    fn preview_camera_targets_volume_center() {
        let volume = demo_volume();
        let camera = camera_for_state(&volume, &VolumeViewState::default());
        let center = volume.world_bounds().center();
        assert!((camera.focal_point() - center).length() < 1e-6);
        assert!(camera.distance() > volume.world_bounds().diagonal());
    }

    #[test]
    fn preview_params_use_composite_rendering() {
        let volume = demo_volume();
        let params = default_render_params(&volume, &VolumeViewState::default());
        assert!(matches!(params.blend_mode, BlendMode::Composite));
        assert!(params.step_size_factor > 0.0);
        assert!(params.opacity_tf.len() >= 4);
        assert!(params.color_tf.len() >= 2);
    }

    #[test]
    fn ct_like_ranges_select_ct_preset() {
        assert!(looks_ct_like(-1024.0, 3071.0));
        assert!(!looks_ct_like(0.0, 255.0));
    }

    #[test]
    fn interactive_render_uses_stronger_downsample_for_large_previews() {
        assert_eq!(interactive_downsample_factor(320, 240), 2);
        assert_eq!(interactive_downsample_factor(640, 480), 2);
    }

    #[test]
    fn volume_view_state_orbit_and_zoom_clamp() {
        let mut state = VolumeViewState::default();
        state.orbit(10.0, 200.0);
        state.zoom_by(100.0);
        // Quaternion orientation changes (no clamp), zoom still clamped at 8.0.
        assert_ne!(state.orientation, DQuat::IDENTITY);
        assert_eq!(state.zoom, 8.0);
    }

    #[test]
    fn transfer_window_defaults_to_soft_tissue_for_ct() {
        let mut state = VolumeViewState::default();
        state.ensure_transfer_window(-1024.0, 3071.0);
        assert_eq!(state.transfer_window(-1024.0, 3071.0), (90.0, 700.0));
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
    fn slice_preview_state_clamps_scroll_to_volume_bounds() {
        let bounds = Aabb::new(DVec3::ZERO, DVec3::new(10.0, 20.0, 30.0));
        let mut state = SlicePreviewState::default();
        state.scroll_by(100.0, bounds);
        assert_eq!(state.offset, 15.0);
        state.scroll_by(-100.0, bounds);
        assert_eq!(state.offset, -15.0);
    }

    #[test]
    fn slice_plane_tracks_requested_orientation() {
        let bounds = Aabb::new(DVec3::ZERO, DVec3::new(10.0, 20.0, 30.0));
        let mut state = SlicePreviewState::default();
        state.set_mode(SlicePreviewMode::Coronal);
        state.offset = 4.0;
        let plane = slice_plane_for_state(bounds, state);
        assert_eq!(plane.origin.y, 14.0);
        assert_eq!(plane.width, 10.0);
        assert_eq!(plane.height, 30.0);
    }

    #[test]
    fn slice_default_planes_follow_radiology_view_conventions() {
        let bounds = Aabb::new(DVec3::ZERO, DVec3::new(10.0, 20.0, 30.0));

        let mut coronal = SlicePreviewState::default();
        coronal.set_mode(SlicePreviewMode::Coronal);
        let coronal_plane = coronal.slice_plane(bounds);
        assert!(coronal_plane.right.distance(DVec3::X) < 1.0e-6);
        assert!(coronal_plane.up.distance(-DVec3::Z) < 1.0e-6);

        let mut sagittal = SlicePreviewState::default();
        sagittal.set_mode(SlicePreviewMode::Sagittal);
        let sagittal_plane = sagittal.slice_plane(bounds);
        assert!(sagittal_plane.right.distance(DVec3::Y) < 1.0e-6);
        assert!(sagittal_plane.up.distance(-DVec3::Z) < 1.0e-6);
    }

    #[test]
    fn slice_projection_mode_cycles_into_thick_slab() {
        let mut state = SlicePreviewState::default();
        state.cycle_projection_mode(6.0);
        assert_eq!(state.projection_mode, SliceProjectionMode::MaximumIntensity);
        assert_eq!(state.slab_half_thickness, 6.0);
        assert!(state.thick_slab().is_some());
    }

    #[test]
    fn slice_projection_mode_is_remembered_per_mpr_axis() {
        let mut state = SlicePreviewState::default();
        state.cycle_projection_mode(6.0);
        assert_eq!(state.projection_mode, SliceProjectionMode::MaximumIntensity);
        assert_eq!(state.slab_half_thickness, 6.0);

        state.set_mode(SlicePreviewMode::Coronal);
        assert_eq!(state.projection_mode, SliceProjectionMode::Thin);
        assert_eq!(state.slab_half_thickness, 0.0);

        state.cycle_projection_mode(10.0);
        state.cycle_projection_mode(10.0);
        assert_eq!(state.projection_mode, SliceProjectionMode::MinimumIntensity);
        assert_eq!(state.slab_half_thickness, 10.0);

        state.set_mode(SlicePreviewMode::Axial);
        assert_eq!(state.projection_mode, SliceProjectionMode::MaximumIntensity);
        assert_eq!(state.slab_half_thickness, 6.0);
    }

    #[test]
    fn slice_slab_drag_can_enable_and_disable_thick_slab() {
        let mut state = SlicePreviewState::default();

        state.set_slab_half_thickness_from_drag(8.0, 0.5, SliceProjectionMode::MaximumIntensity);
        assert_eq!(state.projection_mode, SliceProjectionMode::MaximumIntensity);
        assert_eq!(state.slab_half_thickness, 8.0);

        state.set_slab_half_thickness_from_drag(0.1, 0.5, SliceProjectionMode::MaximumIntensity);
        assert_eq!(state.projection_mode, SliceProjectionMode::Thin);
        assert_eq!(state.slab_half_thickness, 0.0);
    }

    #[test]
    fn slice_crosshair_defaults_to_volume_center() {
        let bounds = Aabb::new(DVec3::ZERO, DVec3::new(10.0, 20.0, 30.0));
        let state = SlicePreviewState::default();
        assert_eq!(state.crosshair_world(bounds), bounds.center());
    }

    #[test]
    fn slice_center_on_world_updates_offset_for_mode() {
        let bounds = Aabb::new(DVec3::ZERO, DVec3::new(10.0, 20.0, 30.0));
        let mut state = SlicePreviewState::default();
        state.center_on_world(DVec3::new(4.0, 9.0, 23.0), bounds);
        assert_eq!(state.offset, 8.0);

        state.set_mode(SlicePreviewMode::Sagittal);
        state.center_on_crosshair(bounds);
        assert_eq!(state.offset, 1.0);
    }

    #[test]
    fn slice_rotation_keeps_plane_family_orthogonal() {
        let bounds = Aabb::new(DVec3::ZERO, DVec3::new(10.0, 20.0, 30.0));
        let mut axial = SlicePreviewState::default();
        axial.rotate_about_normal(std::f64::consts::FRAC_PI_4, bounds);
        let axial_plane = axial.slice_plane(bounds);
        assert!(axial_plane.right.distance(DVec3::X) < 1.0e-6);
        assert!(axial_plane.up.distance(DVec3::Y) < 1.0e-6);

        let mut sagittal = axial;
        sagittal.set_mode(SlicePreviewMode::Sagittal);
        sagittal.center_on_crosshair(bounds);
        let sagittal_plane = sagittal.slice_plane(bounds);
        assert!(sagittal_plane.normal().dot(axial_plane.normal()).abs() < 1.0e-6);
        assert!(sagittal_plane.normal().distance(DVec3::X) > 1.0e-3);
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

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn benchmark_volume_preview_paths() {
        let mut renderer = VolumePreviewRenderer::new().expect("GPU renderer");
        let prepared = PreparedVolume {
            volume: synthetic_ct_volume(),
            cache_key: "synthetic-ct-benchmark".to_string(),
        };
        let mut view_state = VolumeViewState::default();
        view_state.ensure_transfer_window(-1024.0, 3071.0);

        let cold_start = Instant::now();
        let _ = renderer
            .render_prepared_preview(&prepared, &view_state, 640, 480, false)
            .expect("cold render");
        let cold = cold_start.elapsed();

        let full_iterations = 4u32;
        let full_start = Instant::now();
        for _ in 0..full_iterations {
            let _ = renderer
                .render_prepared_preview(&prepared, &view_state, 640, 480, false)
                .expect("full render");
        }
        let full_average = full_start.elapsed() / full_iterations;

        let interactive_iterations = 4u32;
        let interactive_start = Instant::now();
        for _ in 0..interactive_iterations {
            let _ = renderer
                .render_prepared_preview(&prepared, &view_state, 640, 480, true)
                .expect("interactive render");
        }
        let interactive_average = interactive_start.elapsed() / interactive_iterations;

        eprintln!(
            "DVR benchmark cold={cold:?} full_avg={full_average:?} interactive_avg={interactive_average:?}"
        );
        assert!(
            interactive_average < full_average,
            "interactive preview should outperform full-quality preview"
        );
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn rendered_coronal_and_sagittal_place_superior_at_top() {
        let mut renderer = VolumePreviewRenderer::new().expect("GPU renderer");
        let prepared = PreparedVolume {
            volume: z_gradient_volume(),
            cache_key: "z-gradient-mpr".to_string(),
        };
        let mut coronal = SlicePreviewState::default();
        coronal.set_mode(SlicePreviewMode::Coronal);
        coronal.transfer_center_hu = Some(1500.0);
        coronal.transfer_width_hu = Some(4000.0);

        let mut sagittal = coronal;
        sagittal.set_mode(SlicePreviewMode::Sagittal);

        let coronal_image = renderer
            .render_prepared_slice_preview(&prepared, &coronal, 64, 64, false)
            .expect("coronal render");
        let sagittal_image = renderer
            .render_prepared_slice_preview(&prepared, &sagittal, 64, 64, false)
            .expect("sagittal render");

        let coronal_top = average_row_luma(&coronal_image, 0);
        let coronal_bottom = average_row_luma(&coronal_image, coronal_image.height - 1);
        let sagittal_top = average_row_luma(&sagittal_image, 0);
        let sagittal_bottom = average_row_luma(&sagittal_image, sagittal_image.height - 1);

        assert!(
            coronal_top > coronal_bottom,
            "expected superior-at-top coronal render, got top={coronal_top} bottom={coronal_bottom}"
        );
        assert!(
            sagittal_top > sagittal_bottom,
            "expected superior-at-top sagittal render, got top={sagittal_top} bottom={sagittal_bottom}"
        );
    }
}
