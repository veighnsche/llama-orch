# defineModel

**Define reusable model configurations with defaults**

```typescript
import { defineModel } from '@llama-orch/utils/model';

// Define model with defaults
const llama = defineModel({
  name: 'llama-3.1-8b-instruct',
  maxTokens: 8192,
  temperature: 0.7,
  topP: 0.9,
});

// Use model
const response = await llama.invoke('Hello, world!');

// Override defaults
const response = await llama.invoke('Write a haiku', {
  temperature: 0.9,
  maxTokens: 50,
});

// Multiple models
const fast = defineModel({ name: 'llama-3.1-8b-instruct', temperature: 0.0 });
const creative = defineModel({ name: 'llama-3.1-70b-instruct', temperature: 0.9 });
```
