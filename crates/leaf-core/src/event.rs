//! Application-wide event bus for cross-crate communication.
//!
//! Uses tokio broadcast channels so multiple subscribers can react
//! to events like study imports, retrieval progress, and settings changes.

use crate::domain::{SeriesUid, StudyUid};
use tokio::sync::broadcast;

/// Events that flow through the application event bus.
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// A study was imported or updated in the local database.
    StudyIndexed { study_uid: StudyUid },

    /// A study retrieval started from a remote PACS.
    RetrievalStarted {
        study_uid: StudyUid,
        total_instances: u32,
    },

    /// Progress update during retrieval.
    RetrievalProgress {
        study_uid: StudyUid,
        completed: u32,
        total: u32,
    },

    /// Retrieval completed.
    RetrievalCompleted { study_uid: StudyUid },

    /// Retrieval failed.
    RetrievalFailed {
        study_uid: StudyUid,
        reason: String,
    },

    /// User requested to open a study in the viewer.
    OpenStudy { study_uid: StudyUid },

    /// User selected a series in the viewer.
    SeriesSelected { series_uid: SeriesUid },

    /// Configuration was changed and saved.
    ConfigChanged,
}

/// Central event bus for the application.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: broadcast::Sender<AppEvent>,
}

impl EventBus {
    /// Create a new event bus with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Publish an event to all subscribers.
    pub fn publish(&self, event: AppEvent) {
        // Ignore error if no subscribers are listening.
        let _ = self.sender.send(event);
    }

    /// Subscribe to receive events.
    pub fn subscribe(&self) -> broadcast::Receiver<AppEvent> {
        self.sender.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(256)
    }
}
