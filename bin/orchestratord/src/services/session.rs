//! Session service
//!
//! Metadata-only: the orchestrator does not store conversation content (no prompts/messages/outputs).
//! Session state tracks TTL, turns, and engine KV/cache/budgets metadata for scheduling/observability.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::ports::clock::Clock;
use crate::state::SessionInfo;

#[derive(Clone)]
pub struct SessionService<C: Clock + 'static> {
    sessions: Arc<Mutex<HashMap<String, SessionInfo>>>,
    clock: Arc<C>,
}

impl<C: Clock + 'static> SessionService<C> {
    pub fn new(sessions: Arc<Mutex<HashMap<String, SessionInfo>>>, clock: Arc<C>) -> Self {
        Self { sessions, clock }
    }

    pub fn get_or_create(&self, id: &str) -> SessionInfo {
        let mut guard = self.sessions.lock().unwrap();
        let entry = guard.entry(id.to_string()).or_insert_with(|| SessionInfo {
            ttl_ms_remaining: 600_000,
            turns: 0,
            kv_bytes: 0,
            kv_warmth: false,
            tokens_budget_remaining: 0,
            time_budget_remaining_ms: 600_000,
            cost_budget_remaining: 0.0,
        });
        entry.clone()
    }

    /// Decrement TTL using provided now_ms (if None, pull from clock). Returns current entry if it still exists.
    pub fn tick(&self, id: &str, now_ms: Option<u64>) -> Option<SessionInfo> {
        let _now = now_ms.unwrap_or_else(|| self.clock.now_ms());
        let mut guard = self.sessions.lock().unwrap();
        if let Some(s) = guard.get_mut(id) {
            // For now decrement fixed 100ms per tick; replace with last_seen bookkeeping later.
            s.ttl_ms_remaining = (s.ttl_ms_remaining - 100).max(0);
            if s.ttl_ms_remaining <= 0 {
                guard.remove(id);
                return None;
            }
            return Some(s.clone());
        }
        None
    }

    pub fn delete(&self, id: &str) {
        let mut guard = self.sessions.lock().unwrap();
        guard.remove(id);
    }

    /// Increment turns accounting (called when a task is accepted for a session)
    pub fn note_turn(&self, id: &str) -> Option<SessionInfo> {
        let mut guard = self.sessions.lock().unwrap();
        if let Some(s) = guard.get_mut(id) {
            s.turns += 1;
            return Some(s.clone());
        }
        None
    }
}
