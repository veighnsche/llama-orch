//! Artifact service

use crate::state::AppState;

pub fn put(state: &AppState, doc: serde_json::Value) -> anyhow::Result<String> {
    // Persist via configured store
    let id = state.artifact_store.put(doc.clone())?;
    // Maintain in-memory map for compatibility/tests
    if let Ok(mut m) = state.artifacts.lock() {
        m.insert(id.clone(), doc);
    }
    Ok(id)
}

pub fn get(state: &AppState, id: &str) -> anyhow::Result<Option<serde_json::Value>> {
    // Check configured store first
    if let Some(v) = state.artifact_store.get(&id.to_string())? {
        return Ok(Some(v));
    }
    // Fallback to in-memory
    let guard = state.artifacts.lock().unwrap();
    Ok(guard.get(id).cloned())
}
