use serde_json::json;

#[test]
fn inline_snapshot_capabilities() {
    let caps = json!({
        "api_version": "1.0.0",
        "engines": [
            {"engine":"llamacpp","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"vllm","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"tgi","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"triton","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}}
        ]
    });
    insta::assert_yaml_snapshot!(caps, @r###"
---
api_version: 1.0.0
engines:
  - ctx_max: 32768
    engine: llamacpp
    features: {}
    rate_limits: {}
    supported_workloads:
      - completion
      - embedding
      - rerank
  - ctx_max: 32768
    engine: vllm
    features: {}
    rate_limits: {}
    supported_workloads:
      - completion
      - embedding
      - rerank
  - ctx_max: 32768
    engine: tgi
    features: {}
    rate_limits: {}
    supported_workloads:
      - completion
      - embedding
      - rerank
  - ctx_max: 32768
    engine: triton
    features: {}
    rate_limits: {}
    supported_workloads:
      - completion
      - embedding
      - rerank
"###);
}

#[test]
fn inline_snapshot_replicasets() {
    // Minimal enriched payload example mirroring orchestrator's list_replicasets
    let sets = json!([
        {"id":"pool0-llamacpp","engine":"llamacpp","load":0.0,"slots_total":1,"slots_free":1,"slo":{}},
        {"id":"pool0-vllm","engine":"vllm","load":0.0,"slots_total":1,"slots_free":1,"slo":{}}
    ]);
    insta::assert_yaml_snapshot!(sets, @r###"
---
- engine: llamacpp
  id: pool0-llamacpp
  load: 0
  slo: {}
  slots_free: 1
  slots_total: 1
- engine: vllm
  id: pool0-vllm
  load: 0
  slo: {}
  slots_free: 1
  slots_total: 1
"###);
}
