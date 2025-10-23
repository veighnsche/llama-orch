# rbee-openai-adapter

**OpenAI-compatible API adapter for rbee queen**

`bin/15_queen_rbee_crates/rbee-openai-adapter` — Provides OpenAI-compatible HTTP endpoints that translate to rbee operations.

---

## Purpose

Enable existing applications built for the OpenAI API to work with rbee without modification. This adapter translates OpenAI API calls to rbee's internal Operation types.

**Use case:** Point your OpenAI-compatible app to `https://queenbee.home.arpa/openai/v1/*` and it just works.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  External Application (OpenAI SDK)                          │
│  - Python openai library                                    │
│  - JavaScript openai package                                │
│  - Any OpenAI-compatible client                             │
└─────────────────────────────────────────────────────────────┘
                      ↓ HTTP
                      ↓ POST /openai/v1/chat/completions
┌─────────────────────────────────────────────────────────────┐
│  rbee-openai-adapter (Translation Layer)                    │
│  ├─ Parse OpenAI request format                             │
│  ├─ Map to rbee Operation::Infer                            │
│  ├─ Submit to queen's job router                            │
│  └─ Transform response to OpenAI format                     │
└─────────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  queen-rbee (job_router.rs)                                 │
│  - Handles Operation::Infer                                 │
│  - Routes to appropriate worker                             │
│  - Returns SSE stream                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Supported Endpoints

### Chat Completions

**POST /openai/v1/chat/completions**

OpenAI's primary chat endpoint. Supports both streaming and non-streaming.

**Request:**
```json
{
  "model": "llama-3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": true
}
```

**Response (streaming):**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3-8b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3-8b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3-8b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]
```

### Model Management

**GET /openai/v1/models**

List available models in OpenAI format.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3-8b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "rbee"
    }
  ]
}
```

**GET /openai/v1/models/{model}**

Get details for a specific model.

---

## Usage

### In Python (OpenAI SDK)

```python
from openai import OpenAI

# Point to your rbee queen
client = OpenAI(
    base_url="https://queenbee.home.arpa/openai",
    api_key="not-needed"  # rbee doesn't require API keys (yet)
)

# Use exactly like OpenAI
response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### In JavaScript (OpenAI SDK)

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://queenbee.home.arpa/openai',
  apiKey: 'not-needed',
});

const stream = await client.chat.completions.create({
  model: 'llama-3-8b',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello!' }
  ],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

### In Rust (async-openai)

```rust
use async_openai::{Client, types::*};

let client = Client::new()
    .with_base_url("https://queenbee.home.arpa/openai");

let request = CreateChatCompletionRequestArgs::default()
    .model("llama-3-8b")
    .messages(vec![
        ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: "You are a helpful assistant.".to_string(),
                ..Default::default()
            }
        ),
        ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: "Hello!".to_string(),
                ..Default::default()
            }
        ),
    ])
    .build()?;

let response = client.chat().create(request).await?;
println!("{}", response.choices[0].message.content);
```

---

## Integration with queen-rbee

### Mounting the Router

```rust
// In queen-rbee/src/main.rs
use rbee_openai_adapter::create_openai_router;

let app = Router::new()
    // rbee native API
    .nest("/v1", rbee_router)
    
    // OpenAI-compatible API
    .nest("/openai", create_openai_router())
    
    // Health check
    .route("/health", get(health_check));
```

### Request Flow

1. **External app** sends OpenAI-formatted request to `/openai/v1/chat/completions`
2. **Adapter** parses OpenAI request
3. **Adapter** extracts prompt from messages array
4. **Adapter** creates `Operation::Infer { prompt, model, ... }`
5. **Adapter** submits to queen's job router
6. **Queen** processes inference (same as rbee-keeper)
7. **Adapter** transforms SSE stream to OpenAI format
8. **External app** receives OpenAI-formatted response

---

## Implementation Roadmap

### Phase 1: Research & Design (8-12 hours)

**Goal:** Understand OpenAI API specification

**Tasks:**
- [ ] Read OpenAI API documentation thoroughly
- [ ] Document all request/response formats
- [ ] Identify required vs optional fields
- [ ] Map OpenAI parameters to rbee parameters
- [ ] Design error code mapping
- [ ] Create test cases

**Resources:**
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Models](https://platform.openai.com/docs/api-reference/models)

### Phase 2: Core Implementation (12-16 hours)

**Goal:** Implement chat completions endpoint

**Tasks:**
- [ ] Implement `ChatCompletionRequest` parsing
- [ ] Extract prompt from messages array (handle system/user/assistant roles)
- [ ] Map OpenAI model names to rbee model IDs
- [ ] Create `Operation::Infer` from OpenAI request
- [ ] Submit to queen's job router
- [ ] Handle non-streaming responses
- [ ] Handle streaming responses (SSE → OpenAI chunks)
- [ ] Implement error handling and mapping

**Key Challenges:**
- **Message format:** OpenAI uses array of messages, rbee uses single prompt
  - Solution: Concatenate messages with role prefixes
- **Streaming format:** OpenAI uses specific SSE chunk format
  - Solution: Transform rbee SSE events to OpenAI chunks
- **Model naming:** OpenAI uses model IDs like "gpt-4", rbee uses HuggingFace IDs
  - Solution: Model name mapping table

### Phase 3: Model Endpoints (4-6 hours)

**Goal:** Implement model listing and details

**Tasks:**
- [ ] Implement `GET /openai/v1/models`
- [ ] Query rbee model catalog
- [ ] Transform to OpenAI `ModelInfo` format
- [ ] Implement `GET /openai/v1/models/{model}`
- [ ] Handle model not found errors

### Phase 4: Testing & Validation (8-12 hours)

**Goal:** Ensure compatibility with OpenAI SDKs

**Tasks:**
- [ ] Test with Python openai library
- [ ] Test with JavaScript openai package
- [ ] Test with Rust async-openai
- [ ] Test streaming and non-streaming
- [ ] Test error cases
- [ ] Create integration tests
- [ ] Document compatibility notes

**Total Effort:** 32-46 hours

---

## Parameter Mapping

### OpenAI → rbee

| OpenAI Parameter | rbee Parameter | Notes |
|------------------|----------------|-------|
| `model` | `model` | May need name mapping |
| `messages` | `prompt` | Concatenate with role prefixes |
| `temperature` | `temperature` | Direct mapping |
| `top_p` | `top_p` | Direct mapping |
| `max_tokens` | `max_tokens` | Direct mapping |
| `stream` | N/A | Controls response format |
| `seed` | `seed` | Direct mapping |
| `stop` | `stop_sequences` | Direct mapping |

### Message Concatenation

**OpenAI format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}
```

**Converted to rbee prompt:**
```
<|system|>
You are a helpful assistant.

