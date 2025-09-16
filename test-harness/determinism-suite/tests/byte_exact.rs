use test_harness_determinism_suite::{
    generate_deterministic_sse, parse_sse_transcript, snapshot_from_events,
};

fn seeds() -> Vec<u64> {
    (0u64..64u64).collect()
}

// OC-CORE-1030: Determinism invariants — same engine version and same seed should yield byte-exact streams.
#[test]
fn two_replicas_same_engine_same_seed_are_byte_exact() {
    let engine = "llamacpp:v1.0.0";
    let seed = 42u64;
    let s1 = generate_deterministic_sse(engine, seed, 32);
    let s2 = generate_deterministic_sse(engine, seed, 32);
    assert_eq!(s1, s2, "replicas must be byte-exact for same engine+seed");

    let e1 = parse_sse_transcript(&s1);
    let e2 = parse_sse_transcript(&s2);
    let snap1 = snapshot_from_events(engine, seed, &e1);
    let snap2 = snapshot_from_events(engine, seed, &e2);
    assert_eq!(snap1.tokens_first_32, snap2.tokens_first_32);
    assert_eq!(snap1.total_tokens, snap2.total_tokens);
}

// OC-CORE-1030: Determinism is not assumed across engine versions.
#[test]
fn different_engine_versions_not_byte_exact() {
    let seed = 42u64;
    let s1 = generate_deterministic_sse("llamacpp:v1.0.0", seed, 32);
    let s2 = generate_deterministic_sse("llamacpp:v1.0.1", seed, 32);
    assert_ne!(
        s1, s2,
        "engine version change should alter stream determinism domain"
    );
}

// OC-CORE-1030: Different seeds should change the stream tokens for a fixed engine.
#[test]
fn different_seeds_change_tokens() {
    let engine = "tgi:v2.3.0";
    let s1 = generate_deterministic_sse(engine, 1, 16);
    let s2 = generate_deterministic_sse(engine, 2, 16);
    let e1 = parse_sse_transcript(&s1);
    let e2 = parse_sse_transcript(&s2);
    let snap1 = snapshot_from_events(engine, 1, &e1);
    let snap2 = snapshot_from_events(engine, 2, &e2);
    assert_ne!(snap1.tokens_first_32, snap2.tokens_first_32);
}

#[test]
fn seeds_count_is_64_guard() {
    assert_eq!(seeds().len(), 64);
}

#[test]
#[ignore]
fn byte_exact_streams_placeholder() {
    // Placeholder: in the future, fetch streams from two replicas and compare byte-exactness.
    let sample = "event: started\n\n\
                  data: {\"queue_position\":3,\"predicted_start_ms\":420}\n\n\
                  event: token\n\n\
                  data: {\"t\":\"Hello\",\"i\":0}\n\n\
                  event: end\n\n\
                  data: {\"tokens_out\":1,\"decode_ms\":100}\n\n";
    let evs = parse_sse_transcript(sample);
    let snap = snapshot_from_events("llamacpp", 123, &evs);
    assert_eq!(snap.tokens_first_32.len(), 1);
}
