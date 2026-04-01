//! pacsleaf — Pure Rust Medical Imaging Viewer
//!
//! Companion viewer for pacsnode. Provides a fast, modern, dark-themed
//! clinical workstation for radiologists.

use anyhow::Result;
use leaf_core::config::{config_dir, AppConfig};
use leaf_db::imagebox::Imagebox;
use slint::ComponentHandle;
use tracing::info;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    // Initialize logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("pacsleaf v{} starting", env!("CARGO_PKG_VERSION"));

    // Load configuration.
    let config_path = config_dir().join("pacsleaf.toml");
    let config = AppConfig::load(&config_path).unwrap_or_default();

    // Open the local imagebox database.
    let db_path = config.resolved_db_path();
    let _imagebox = Imagebox::open(&db_path)?;

    // Create the event bus.
    let _event_bus = leaf_core::event::EventBus::default();

    // Launch the Slint UI.
    let browser = leaf_ui::StudyBrowserWindow::new()?;

    // Wire up callbacks.
    let browser_weak = browser.as_weak();
    browser.on_search(move |name, id, accession| {
        let _browser = browser_weak.upgrade().unwrap();
        info!(
            "Search: name={}, id={}, accession={}",
            name, id, accession
        );
        // TODO: Query local DB and/or remote PACS
    });

    browser.on_open_study(move |study_uid| {
        info!("Opening study: {}", study_uid);
        // TODO: Launch StudyViewerWindow with this study
    });

    browser.on_import_files(move || {
        info!("Import files requested");
        // TODO: Open file dialog and import DICOM files
    });

    // Run the application.
    browser.run()?;

    info!("pacsleaf shutting down");
    Ok(())
}
