# 🍃 pacsleaf

**Pure Rust Medical Imaging Viewer** — the companion product to [pacsnode](../pacsnode/).

> ⚠️ **Not for clinical use.** This software is not validated for diagnostic or therapeutic purposes.

## Features

- **Ultrafast** — Zero-copy database (redb), GPU-accelerated rendering (wgpu), parallel DICOM decode
- **Pure Rust** — Single binary, no C/C++ runtime dependencies
- **Dark-first UI** — Modern, lean design via Slint, inspired by OHIF and KPACS-neo
- **Multi-protocol** — DIMSE (dicom-toolkit-rs) and DICOMweb for maximum PACS compatibility
- **Volume rendering** — MPR, MIP, DVR via volren-rs with real-time GPU raycasting
- **Clinical tools** — Window/level, measurements, ROI analysis, annotations, color LUTs

## Architecture

```
pacsleaf/
├── crates/
│   ├── leaf-core/       Domain types, config, event bus
│   ├── leaf-db/         redb-backed local imagebox
│   ├── leaf-dicom/      dicom-toolkit-rs bridge
│   ├── leaf-net/        DIMSE + DICOMweb clients
│   ├── leaf-render/     2D & volume rendering (volren-rs)
│   ├── leaf-tools/      Measurements & annotations
│   ├── leaf-ui/         Slint UI definitions
│   └── leaf-app/        Binary entry point
└── ui/
    ├── theme/           Color palette, typography
    ├── components/      Reusable widgets
    ├── windows/         Study browser, viewer
    └── assets/          Icons, cursors
```

## Quick Start

```bash
# Build and run
cargo run --release -p leaf-app

# Run with debug logging
RUST_LOG=debug cargo run -p leaf-app
```

## Configuration

Configuration is stored in `~/.config/pacsleaf/pacsleaf.toml` (Linux/macOS) or `%APPDATA%/pacsleaf/pacsleaf.toml` (Windows).

```toml
[general]
ae_title = "PACSLEAF"
port = 11114

[database]
cache_size_mb = 512

[display]
default_lut = "grayscale"
default_layout = "1x1"

[[nodes]]
name = "pacsnode-local"
host = "127.0.0.1"
port = 4242
ae_title = "PACSNODE"
protocol = "both"
dicomweb_url = "http://localhost:3000/wado"
```

## Dependencies

- [dicom-toolkit-rs](../dicom-toolkit-rs/) — DICOM parsing, DIMSE networking, image codecs
- [volren-rs](../volren-rs/) — GPU-accelerated volume rendering
- [Slint](https://slint.dev/) — Cross-platform GUI framework with wgpu integration
- [redb](https://github.com/cberner/redb) — Embedded database

## License

MIT OR Apache-2.0
