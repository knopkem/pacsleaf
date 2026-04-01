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
```

## Dependencies

- [dicom-toolkit-rs](../dicom-toolkit-rs/) — DICOM parsing, DIMSE networking, image codecs
- [volren-rs](../volren-rs/) — GPU-accelerated volume rendering
- [Slint](https://slint.dev/) — Cross-platform GUI framework with wgpu integration
- [redb](https://github.com/cberner/redb) — Embedded database

## License

MIT OR Apache-2.0
