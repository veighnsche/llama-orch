use test_harness_determinism_suite::{parse_sse_transcript, snapshot_from_events};

fn seeds() -> Vec<u64> {
    (0u64..64u64).collect()
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
