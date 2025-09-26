use super::*;

#[test]
fn applies_defaults_when_fields_missing() {
    let p = Params { temperature: None, top_p: None, max_tokens: None, seed: None };
    let out = run(p);
    assert_eq!(out.temperature, Some(0.7));
    assert_eq!(out.top_p, Some(1.0));
    assert_eq!(out.max_tokens, Some(1024));
    assert_eq!(out.seed, None);
}

#[test]
fn clamps_values_to_bounds() {
    // Below minimums
    let p1 = Params { temperature: Some(-1.0), top_p: Some(-0.5), max_tokens: Some(0), seed: None };
    let out1 = run(p1);
    assert_eq!(out1.temperature, Some(0.0));
    assert_eq!(out1.top_p, Some(0.0));
    assert_eq!(out1.max_tokens, Some(0));

    // Above maximums
    let p2 = Params {
        temperature: Some(9.9),
        top_p: Some(9.9),
        max_tokens: Some(1_000_000),
        seed: None,
    };
    let out2 = run(p2);
    assert_eq!(out2.temperature, Some(2.0));
    assert_eq!(out2.top_p, Some(1.0));
    assert_eq!(out2.max_tokens, Some(1_000_000)); // no clamp rule for max_tokens value itself
}

#[test]
fn passes_through_valid_explicit_values() {
    let p =
        Params { temperature: Some(1.1), top_p: Some(0.95), max_tokens: Some(256), seed: Some(42) };
    let out = run(p.clone());
    assert_eq!(out.temperature, Some(1.1));
    assert_eq!(out.top_p, Some(0.95));
    assert_eq!(out.max_tokens, Some(256));
    assert_eq!(out.seed, Some(42));
}
