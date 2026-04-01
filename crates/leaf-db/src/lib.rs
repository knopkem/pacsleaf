//! Local imagebox database for pacsleaf, backed by redb.
//!
//! Provides zero-copy metadata queries for ultrafast study browsing,
//! with async wrappers for non-blocking operation.

pub mod imagebox;
pub mod schema;
