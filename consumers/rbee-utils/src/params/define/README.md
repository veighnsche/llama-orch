# defineParams

**Define reusable inference parameter sets**

```typescript
import { defineParams } from '@llama-orch/utils/params';

// Define parameter set
const creative = defineParams({
  temperature: 0.9,
  topP: 0.95,
  topK: 50,
});

const deterministic = defineParams({
  temperature: 0.0,
  seed: 42,
});

// Use with invoke
const response = await invoke({
  prompt: 'Write a story',
  ...creative,
});

// Merge parameter sets
const params = { ...creative, maxTokens: 100 };
```
