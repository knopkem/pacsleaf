# Copilot instructions for `pacsleaf`

**Pure Rust Medical Imaging Viewer** — the companion product to [pacsnode](https://github.com/knopkem/pacsnode).

## Performance

UI responsiveness is a top priority. Every interaction — tool switching, button clicks, drag operations — must feel instant (< 16 ms). Avoid rebuilding Slint models, recalculating layouts, or re-rendering images in response to lightweight state changes. Defer heavy work (GPU renders, model rebuilds) to when it's actually needed. Profile before optimizing, but never ship sluggish UI.

---

## Code Quality

- Write **production-ready Rust** — no placeholder logic, no `todo!()` left in non-test code, no `unwrap()` or `expect()` outside of tests or `main` startup validation where a panic is acceptable.
- Prefer `?` for error propagation. Define domain-specific error types with `thiserror`. Never use `anyhow` in library crates; `anyhow` is acceptable only in binary entry points.
- All public items must have doc comments (`///`). Include at least one `# Example` block for non-trivial public APIs.
- No `clippy` warnings — code must pass `cargo clippy -- -D warnings` clean. Apply `#[allow(...)]` only when genuinely necessary and always with a comment explaining why.
- Format all code with `rustfmt` (default settings). Never submit unformatted code.
- Avoid `unsafe` unless interfacing with C FFI (e.g., OpenJPEG). Every `unsafe` block must have a `// SAFETY:` comment explaining the invariants upheld.

---

## Rust Patterns

Apply idiomatic Rust patterns consistently:

- **Newtype pattern** for domain identifiers (e.g., `StudyUid(String)`, `SeriesUid(String)`) — prevents mixing up UIDs at the type level.
- **Builder pattern** for structs with many optional fields (e.g., query builders, config structs). Implement via a dedicated `XxxBuilder` struct with a consuming `build() -> Result<Xxx>`.
- **Typestate pattern** for protocol state machines (e.g., DIMSE association lifecycle: `Association<Unassociated>` → `Association<Established>`).
- **`From`/`Into`/`TryFrom`/`TryInto`** for all conversions between domain types and external types (DICOM elements, database rows, API DTOs).
- **`Display` + `Error`** implementations on all error types.
- **`Default`** on config and option structs where zero-value defaults are meaningful.
- Prefer **`Arc<dyn Trait>`** for shared, injectable dependencies (`MetadataStore`, `BlobStore`) — enables testing with mocks.
- Use **`tokio::sync`** primitives (`RwLock`, `Mutex`, `broadcast`, `mpsc`) over `std::sync` in async code.
- Leverage **`tower` middleware** (tracing, timeout, rate-limit) for Axum routes rather than duplicating cross-cutting logic in handlers.
- Prefer **`bytes::Bytes`** for zero-copy binary data passing between components (DICOM pixel data, multipart bodies).

---

## Error Handling

- Library crates define their own `Error` enum with `thiserror`.
- Errors must be meaningful to the caller — wrap lower-level errors with context using `#[from]` or `.map_err(|e| Error::StoreFailed { source: e, uid: uid.to_string() })`.
- Never silently swallow errors. Log with `tracing::error!` at the boundary where you decide not to propagate.

---

## Async & Concurrency

- All I/O is async. Blocking operations (file I/O, CPU-heavy codec work) must be dispatched via `tokio::task::spawn_blocking`.
- Avoid holding locks across `.await` points. Prefer scoped lock guards or restructure code to release before awaiting.
- Cancellation safety: document any `async fn` that is NOT cancellation-safe with a `# Cancellation Safety` section in its doc comment.

---

## Logging & Tracing

- Use `tracing` spans and events throughout, not `println!` or `log!`.
- Instrument all service-layer functions with `#[tracing::instrument(skip(self), err)]`.
- Include structured fields on spans: `study_uid`, `series_uid`, `instance_uid`, `ae_title` as appropriate.
- Log at the right level: `trace!` for per-frame/per-tag operations, `debug!` for per-instance, `info!` for per-study and connection lifecycle, `warn!` for recoverable issues, `error!` for failures.


## High-level architecture

This is an 8-crate Rust workspace with a thin Slint UI layer and most runtime orchestration concentrated in `crates/leaf-app/src/main.rs`.

- `leaf-core` defines shared domain/config/error types such as `StudyInfo`, `SeriesInfo`, `InstanceInfo`, `LeafError`, and PACS node config types.
- `leaf-db` is the local imagebox repository built on `redb`. It stores study/series/instance records as JSON strings and maintains separate index tables for patient → studies, study → series, and series → instances.
- `leaf-dicom` is the bridge to `dicom-toolkit-rs`. It handles metadata extraction, pixel decoding, and volume assembly.
- `leaf-net` wraps networking: DIMSE in `dimse.rs` and DICOMweb in `dicomweb.rs`.
- `leaf-render` contains rendering-oriented state and GPU-facing code, but the current interactive 2D viewer behavior is still largely driven from `leaf-app`.
- `leaf-tools` contains measurement and annotation types/computation such as `Measurement`, `MeasurementKind`, and measurement math.
- `leaf-ui` compiles `ui/app.slint` and exposes the generated Slint components plus the `image_from_rgba8` helper used to display decoded frames.
- `leaf-app` wires everything together: startup, import flow, browser search, local/remote viewer opening, slice sorting, viewer input handling, and in-memory measurement state.

The GUI is split between `ui/windows/study-browser.slint` and `ui/windows/study-viewer.slint`, with shared styling in `ui/theme/theme.slint` and small reusable components in `ui/components/`.

Two sibling repositories are part of the effective architecture, not optional extras:

- `../dicom-toolkit-rs` provides the DICOM parser/image/network crates used throughout the workspace.
- `../volren-rs` provides the `volren-core` and `volren-gpu` path dependencies.

The main local viewer flow is:

1. Import or query studies into `leaf-db::Imagebox`.
2. Open a study from the browser in `leaf-app`.
3. Load series/instances from `Imagebox`.
4. Sort instances into a stack with `sort_instances_for_stack`.
5. Expand instances into frame references.
6. Decode frames through `leaf-dicom::pixel`.
7. Convert RGBA bytes with `leaf_ui::image_from_rgba8`.
8. Push the image and interaction state into `StudyViewerWindow`.

Remote browsing/opening is also orchestrated in `leaf-app`: browser search merges local results with DICOMweb results when network mode is enabled, and remote study rows are encoded as `remote:<node_name>:<study_uid>`.


