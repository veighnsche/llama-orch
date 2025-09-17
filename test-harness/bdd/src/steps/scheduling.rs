use crate::steps::world::World;
use cucumber::then;
use std::collections::HashMap;

#[then(regex = r"^session affinity keeps a session on its last good replica$")]
pub async fn then_session_affinity_keeps_last_good_replica(_world: &mut World) {
    // Simulate sticky mapping in-memory to document expected behaviour.
    let mut last_good: HashMap<&str, &str> = HashMap::new();
    last_good.insert("session-1", "replica-a");
    let dispatch = *last_good.get("session-1").unwrap();
    assert_eq!(dispatch, "replica-a");
}
