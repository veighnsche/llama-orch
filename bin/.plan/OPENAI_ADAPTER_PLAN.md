# OpenAI API Adapter Implementation Plan

**Created by:** TEAM-266  
**Date:** October 23, 2025  
**Status:** Design Phase

---

## 🎯 Mission

Implement OpenAI-compatible API adapter for rbee queen, enabling existing OpenAI applications to work with rbee without modification.

**Goal:** Point any OpenAI SDK to `https://queenbee.home.arpa/openai/v1/*` and it just works.

---

## 📁 Crate Structure

**Location:** `bin/15_queen_rbee_crates/rbee-openai-adapter/`

**Status:** ✅ Stub crate created with:
- ✅ Cargo.toml with dependencies
- ✅ Type definitions (OpenAI API types)
- ✅ Handler stubs (unimplemented)
- ✅ Router structure
- ✅ Error types
- ✅ README with usage examples

---

## 🔄 Request Flow

```
External App → /openai/v1/chat/completions
    ↓
OpenAI Adapter (parse request)
    ↓
Extract prompt from messages array
    ↓
Create Operation::Infer
    ↓
Submit to queen's job_router
    ↓
Transform SSE stream to OpenAI format
    ↓
Return to external app
```

---

## 📋 Implementation Phases

### Phase 1: Research & Design (8-12 hours)

**Goal:** Thoroughly understand OpenAI API specification

**Tasks:**
- [ ] Read [OpenAI API Reference](https://platform.openai.com/docs/api-reference) completely
- [ ] Document all chat completion parameters
- [ ] Document streaming SSE format
- [ ] Document error response format
- [ ] Map OpenAI parameters to rbee parameters
- [ ] Design message-to-prompt conversion strategy
- [ ] Design model name mapping strategy
- [ ] Create test cases for edge cases

**Deliverables:**
- OpenAI API specification summary document
- Parameter mapping table
- Message conversion algorithm
- Test case list

**Key Questions to Answer:**
1. How to handle multi-turn conversations (messages array)?
2. How to map OpenAI model names to HuggingFace IDs?
3. How to handle unsupported parameters gracefully?
4. What's the exact SSE chunk format for streaming?
5. How to handle errors (what error codes does OpenAI use)?

---

### Phase 2: Core Implementation (12-16 hours)

**Goal:** Implement chat completions endpoint (non-streaming first)

#### Step 1: Request Parsing (2-3 hours)

```rust
// src/handlers.rs
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, OpenAIError> {
    // 1. Validate request
    validate_request(&request)?;
    
    // 2. Extract prompt from messages
    let prompt = messages_to_prompt(&request.messages)?;
    
    // 3. Map model name
    let model = map_model_name(&request.model)?;
    
    // 4. Create rbee operation
    let operation = Operation::Infer {
        prompt,
        model,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        seed: request.seed.map(|s| s as i64),
        // ... other parameters
    };
    
    // 5. Submit to queen
    if request.stream {
        handle_streaming(operation, &request).await
    } else {
        handle_non_streaming(operation, &request).await
    }
}
```

**Tasks:**
- [ ] Implement `validate_request()`
- [ ] Implement `messages_to_prompt()` - Convert messages array to single prompt
- [ ] Implement `map_model_name()` - Map OpenAI names to rbee IDs
- [ ] Add error handling for invalid requests

#### Step 2: Message Conversion (3-4 hours)

**Challenge:** OpenAI uses array of messages, rbee uses single prompt.

**Solution:** Concatenate with role markers.

```rust
fn messages_to_prompt(messages: &[ChatMessage]) -> Result<String> {
    let mut prompt = String::new();
    
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|system|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("\n</|system|>\n\n");
            }
            "user" => {
                prompt.push_str("
