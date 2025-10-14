# rbee-utils

**TypeScript utilities for building LLM pipelines with rbee**

`consumers/rbee-utils` — Helper functions and composable utilities for TypeScript/Node.js applications.

---

## What This Library Does

rbee-utils provides **TypeScript utilities** for rbee:

- **File operations** — Read/write files with streaming
- **LLM invocation** — Simplified API for inference
- **Model definitions** — Type-safe model configurations
- **Response extraction** — Parse and extract structured data
- **Prompt management** — Message and thread builders
- **Parameter helpers** — Type-safe parameter definitions

**Used by**: Node.js applications, TypeScript projects

---

## Installation

```bash
npm install @rbee/utils
```

---

## Modules

### File Operations

```typescript
import { FileReader, FileWriter } from '@rbee/utils/fs';

// Read file
const content = await FileReader.read('input.txt');

// Write file
await FileWriter.write('output.txt', content);

// Stream large files
const stream = FileReader.stream('large-file.txt');
for await (const chunk of stream) {
  console.log(chunk);
}
```

### LLM Invocation

```typescript
import { invoke } from '@llama-orch/utils/llm';

const response = await invoke({
  prompt: 'Hello, world!',
  model: 'llama-3.1-8b-instruct',
  maxTokens: 100,
  seed: 42,
});

console.log(response.text);
```

### Model Definitions

```typescript
import { defineModel } from '@llama-orch/utils/model';

const model = defineModel({
  name: 'llama-3.1-8b-instruct',
  maxTokens: 8192,
  temperature: 0.7,
});

const response = await model.invoke('Hello, world!');
```

### Response Extraction

```typescript
import { extractJson, extractCode } from '@llama-orch/utils/orch';

// Extract JSON from response
const data = extractJson(response.text);

// Extract code blocks
const code = extractCode(response.text, 'typescript');
```

### Prompt Management

```typescript
import { Message, Thread } from '@llama-orch/utils/prompt';

// Create message
const message = Message.user('Hello, world!');

// Build thread
const thread = Thread.create()
  .addSystem('You are a helpful assistant')
  .addUser('What is 2+2?')
  .addAssistant('4')
  .addUser('What is 3+3?');

const response = await invoke({
  messages: thread.toMessages(),
  model: 'llama-3.1-8b-instruct',
});
```

### Parameter Helpers

```typescript
import { defineParams } from '@llama-orch/utils/params';

const params = defineParams({
  temperature: 0.7,
  topP: 0.9,
  seed: 42,
  maxTokens: 100,
});

const response = await invoke({
  prompt: 'Hello, world!',
  ...params,
});
```

---

## Testing

### Unit Tests

```bash
# Run all tests
npm test

# Run specific test
npm test -- file_reader
```

---

## Dependencies

### Internal

- `@llama-orch/sdk` — Core SDK

### External

- None (minimal dependencies)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
