# llm/invoke

## Purpose
Send {messages, model_ref, params} to llama-orch via the SDK. Signature simple; internals handle HTTP/streaming.

## Input
```json
{
  "messages": [ { "role": "system", "content": "…" }, { "role": "user", "content": "…" } ],
  "model_ref": { "model_id": "required.model", "engine_id": null, "pool_hint": null },
  "params_ref": { "temperature": 0.2 }
}
```

## Output (raw orchestrator)
Free to match your SDK, but keep it stable for the extractor. Example:
```json
{
  "result": {
    "choices": [ { "text": "final answer…" } ],
    "usage": { "prompt_tokens": 123, "completion_tokens": 456 }
  }
}
```

## Rust surface (invoke.rs)
```rust
use llama_orch_sdk::{OrchestratorClient, Message as SdkMsg, InvokeResult};

pub struct InvokeIn {
    pub messages: Vec<SdkMsg>,
    pub model: crate::model::define::ModelRef,
    pub params: crate::params::define::Params
}
pub struct InvokeOut { pub result: InvokeResult }

pub fn run(client: &OrchestratorClient, input: InvokeIn) -> anyhow::Result<InvokeOut>;
```

## Notes
Keep it blocking/non-streaming for M2; you can add streaming later without breaking the signature (return a unified InvokeResult).
Only depends on SDK; applet does no HTTP itself.
