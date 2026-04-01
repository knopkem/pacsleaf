# 🍃 pacsleaf

**Pure Rust Medical Imaging Viewer** — the companion product to [pacsnode](https://github.com/knopkem/pacsnode).

> ⚠️ **Not for clinical use.** This software is not validated for diagnostic or therapeutic purposes.

## Features

- **Ultrafast** — Zero-copy database (redb), GPU-accelerated rendering (wgpu), parallel DICOM decode (tokio-rs)
- **Pure Rust** — Single binary, no C/C++ runtime dependencies
- **Dark-first UI** — Modern, lean design via Slint
- **Multi-protocol** — DIMSE and DICOMweb support for maximum PACS compatibility
- **Volume rendering** — MPR, MIP, DVR with real-time GPU raycasting
- **Clinical tools** — Window/level, measurements, ROI analysis, annotations, color LUTs

## Quick Start

```bash
# Build and run
cargo run --release -p leaf-app

# Run with debug logging
RUST_LOG=debug cargo run -p leaf-app

# Import one or more local DICOM files/folders before launch
cargo run -p leaf-app -- --import /path/to/study-or-folder --import /path/to/another-folder
```

When the browser is open in `Local` mode, the `Import Folder...` button opens a native folder picker and indexes any DICOM files it finds in the selected directory.

## Testing

1. Launch the app:

```bash
cargo run -p leaf-app
```

2. In `Local` mode, click `Import Folder...` and choose a folder that contains DICOM files.

3. Wait for pacsleaf to index the selected folder.

4. Double-click a local study row to open it in the viewer.

5. In the viewer:

- Use the mouse wheel to scroll slices / frames
- Click series thumbnails on the left to switch series
- Toggle the measurement panel with the `Meas` toolbar button

6. To test remote DICOMweb:

- Open `⚙`
- Enter a node name and DICOMweb base URL in the network drawer
- Click `Save`
- Click `Network`
- Search by patient name / ID / accession
- Double-click a remote result row to open the first rendered frame

## Dependencies

- [dicom-toolkit-rs](../dicom-toolkit-rs/) — DICOM parsing, DIMSE networking, image codecs
- [volren-rs](../volren-rs/) — GPU-accelerated volume rendering
- [Slint](https://slint.dev/) — Cross-platform GUI framework with wgpu integration
- [redb](https://github.com/cberner/redb) — Embedded database

## License

MIT OR Apache-2.0
