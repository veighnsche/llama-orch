//! E2E Haiku harness helpers for OrchQueue v1.

use anyhow::Result;
use contracts_api_types as api;
use tools_openapi_client as client;

pub fn client(base: &str) -> client::Client {
    client::Client::new(base)
}

pub fn build_task(task_id: &str, session_id: &str) -> api::TaskRequest {
    use api::*;
    TaskRequest {
        task_id: task_id.to_string(),
        session_id: session_id.to_string(),
        workload: Workload::Completion,
        model_ref: "repo:NousResearch/Meta-Llama-3-8B-Instruct".to_string(),
        engine: Engine::Llamacpp,
        ctx: 8192,
        priority: Priority::Interactive,
        seed: Some(123456789),
        determinism: Some(DeterminismLevel::Strict),
        sampler_profile_version: Some("v1".to_string()),
        prompt: Some("Write a haiku about GPUs".to_string()),
        inputs: None,
        max_tokens: 64,
        deadline_ms: 30000,
        expected_tokens: Some(64),
        kv_hint: Some(KVHint::Reuse),
    }
}

pub async fn enqueue(base: &str, req: &api::TaskRequest) -> Result<reqwest::Response> {
    Ok(client(base).create_task(req).send().await?)
}

pub async fn stream(base: &str, id: &str) -> Result<reqwest::Response> {
    Ok(client(base).stream_task(id).send().await?)
}

pub async fn scrape_metrics(base: &str) -> Result<String> {
    let url = format!("{}/metrics", base);
    let text = reqwest::Client::new().get(url).send().await?.text().await?;
    Ok(text)
}

/// Very small Prom text parser to map counter/gauge names to a set of label keys seen.
pub fn prom_parse_names_labels(
    prom_text: &str,
) -> std::collections::BTreeMap<String, std::collections::BTreeSet<String>> {
    let mut out: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        Default::default();
    for line in prom_text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // metric_name{label="value",...} number
        if let Some((head, _rest)) = line.split_once(' ') {
            if let Some(start) = head.find('{') {
                let name = &head[..start];
                let labels = &head[start + 1..head.len() - 1];
                let mut keys: std::collections::BTreeSet<String> = Default::default();
                for part in labels.split(',') {
                    if let Some((k, _v)) = part.split_once('=') {
                        keys.insert(k.to_string());
                    }
                }
                out.entry(name.to_string())
                    .or_default()
                    .extend(keys);
            } else {
                out.entry(head.to_string()).or_default();
            }
        }
    }
    out
}

/// Compute token deltas from two Prom snapshots (placeholder using tokens_out_total).
pub fn prom_tokens_out_delta(prev: &str, curr: &str) -> i64 {
    fn parse_total(s: &str, metric: &str) -> i64 {
        let mut sum = 0i64;
        for line in s.lines() {
            let line = line.trim();
            if line.starts_with(metric) {
                if let Some((_head, num)) = line.split_once(' ') {
                    if let Ok(v) = num.trim().parse::<f64>() {
                        sum += v as i64;
                    }
                }
            }
        }
        sum
    }
    parse_total(curr, "tokens_out_total") - parse_total(prev, "tokens_out_total")
}
