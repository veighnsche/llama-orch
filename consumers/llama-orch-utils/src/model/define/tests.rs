use super::*;

#[test]
fn creates_minimal_modelref() {
    let m = run("meta.llama3.1-8b-instruct".to_string(), None, None);
    assert_eq!(m.model_id, "meta.llama3.1-8b-instruct");
    assert!(m.engine_id.is_none());
    assert!(m.pool_hint.is_none());
}

#[test]
fn accepts_optional_engine_and_pool() {
    let m = run(
        "meta.llama3.1-70b-instruct".to_string(),
        Some("llamacpp".to_string()),
        Some("gpu-a100".to_string()),
    );
    assert_eq!(m.model_id, "meta.llama3.1-70b-instruct");
    assert_eq!(m.engine_id.as_deref(), Some("llamacpp"));
    assert_eq!(m.pool_hint.as_deref(), Some("gpu-a100"));
}
