# rbee-openai-adapter

**OpenAI-compatible API adapter for rbee queen**

`bin/15_queen_rbee_crates/rbee-openai-adapter` — Provides OpenAI-compatible HTTP endpoints that translate to rbee operations.

---

## Purpose

Enable existing applications built for the OpenAI API to work with rbee without modification. This adapter translates OpenAI API calls to rbee's internal Operation types.

**Use case:** Point your OpenAI-compatible app to `https://queenbee.home.arpa/openai/v1/*` and it just works.

---

## Status

**STUB CRATE** - Design phase only.

- ✅ Crate structure created
- ✅ Type definitions (OpenAI API types)
- ✅ Handler stubs (unimplemented)
- ✅ Router structure
- ⚠️ Requires 32-46 hours to implement
- ⚠️ Requires extensive OpenAI API research

---

## Compatibility Notes

### What Works (After Implementation)

- ✅ Chat completions (streaming and non-streaming)
- ✅ Model listing
- ✅ Python openai library
- ✅ JavaScript openai package
- ✅ Rust async-openai

### What's Different

- ⚠️ **No API keys required** (rbee doesn't use auth yet)
- ⚠️ **Model names** may differ (HuggingFace IDs vs OpenAI model names)
- ⚠️ **Message format** converted to single prompt
- ⚠️ **Some parameters** may not be supported

### Not Supported

- ❌ Function calling (OpenAI-specific feature)
- ❌ Vision models (not in rbee yet)
- ❌ Embeddings endpoint (different workload)
- ❌ Fine-tuning endpoints (not applicable)

---

## Testing

### Manual Testing

```bash
# Start queen with OpenAI adapter
cargo run --bin queen-rbee

# Test with curl
curl https://queenbee.home.arpa/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Integration Tests

```rust
#[tokio::test]
async fn test_openai_chat_completion() {
    let client = reqwest::Client::new();
    let response = client
        .post("http://localhost:8500/openai/v1/chat/completions")
        .json(&serde_json::json!({
            "model": "llama-3-8b",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }))
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), 200);
}
```

---

## Dependencies

```toml
[dependencies]
axum = "0.7"                    # HTTP server
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rbee-operations = { path = "../../../99_shared_crates/rbee-operations" }
```

---

## References

**OpenAI API Documentation:**
- [API Reference](https://platform.openai.com/docs/api-reference)
- [Chat Completions](https://platform.openai.com/docs/api-reference/chat/create)
- [Models](https://platform.openai.com/docs/api-reference/models)
- [Streaming](https://platform.openai.com/docs/api-reference/streaming)

**Similar Projects:**
- [LocalAI](https://github.com/mudler/LocalAI) - OpenAI-compatible API for local models
- [vLLM](https://github.com/vllm-project/vllm) - Has OpenAI-compatible server
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - HuggingFace's OpenAI-compatible API

---

## Next Steps

1. **Research Phase** - Study OpenAI API spec thoroughly
2. **Implement chat completions** - Core endpoint
3. **Implement streaming** - SSE transformation
4. **Implement model endpoints** - List and get
5. **Test with real SDKs** - Python, JavaScript, Rust
6. **Document compatibility** - What works, what doesn't

**See:** `.arch/OPENAI_ADAPTER_ARCHITECTURE.md` (to be created) for detailed design.

---

**Created by:** TEAM-266  
**Status:** Stub crate - ready for implementation  
**Effort:** 32-46 hours
