use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use orchestratord::services::session::SessionService;
use orchestratord::state::SessionInfo;

#[test]
fn get_or_create_defaults() {
    let svc = SessionService::new(Arc::new(Mutex::new(HashMap::new())), Arc::new(orchestratord::infra::clock::SystemClock::default()));
    let s = svc.get_or_create("s-1");
    assert_eq!(s.ttl_ms_remaining, 600_000);
    assert_eq!(s.turns, 0);
    assert_eq!(s.kv_bytes, 0);
    assert!(!s.kv_warmth);
}

#[test]
fn note_turn_increments_turns() {
    let sessions = Arc::new(Mutex::new(HashMap::new()));
    let svc = SessionService::new(sessions.clone(), Arc::new(orchestratord::infra::clock::SystemClock::default()));
    let _ = svc.get_or_create("s-2");
    let s = svc.note_turn("s-2").unwrap();
    assert_eq!(s.turns, 1);
}

#[test]
fn delete_removes_session() {
    let sessions = Arc::new(Mutex::new(HashMap::new()));
    let svc = SessionService::new(sessions.clone(), Arc::new(orchestratord::infra::clock::SystemClock::default()));
    let _ = svc.get_or_create("s-3");
    svc.delete("s-3");
    let g = sessions.lock().unwrap();
    assert!(!g.contains_key("s-3"));
}

#[test]
fn tick_evicts_when_ttl_reaches_zero() {
    let sessions = Arc::new(Mutex::new(HashMap::new()));
    let svc = SessionService::new(sessions.clone(), Arc::new(orchestratord::infra::clock::SystemClock::default()));
    // Seed custom TTL small enough to evict in one tick
    sessions.lock().unwrap().insert("s-4".into(), SessionInfo { ttl_ms_remaining: 50, ..Default::default() });
    let r = svc.tick("s-4", None);
    assert!(r.is_none());
    assert!(!sessions.lock().unwrap().contains_key("s-4"));
}
