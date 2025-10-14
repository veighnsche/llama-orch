# invoke

**Simplified LLM invocation with type-safe parameters**

```typescript
import { invoke } from '@llama-orch/utils/llm';

// Basic invocation
const response = await invoke({
  prompt: 'Hello, world!',
  model: 'llama-3.1-8b-instruct',
  maxTokens: 100,
});

console.log(response.text);

// With parameters
const response = await invoke({
  prompt: 'Write a haiku',
  model: 'llama-3.1-8b-instruct',
  temperature: 0.7,
  topP: 0.9,
  seed: 42,
  maxTokens: 50,
});

// Streaming
const stream = await invoke.stream({
  prompt: 'Tell me a story',
  model: 'llama-3.1-8b-instruct',
});

for await (const token of stream) {
  process.stdout.write(token);
}
```
