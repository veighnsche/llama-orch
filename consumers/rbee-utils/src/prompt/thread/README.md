# Thread

**Build and manage conversation threads**

```typescript
import { Thread } from '@llama-orch/utils/prompt';

// Create thread
const thread = Thread.create()
  .addSystem('You are a helpful assistant')
  .addUser('What is 2+2?')
  .addAssistant('4')
  .addUser('What is 3+3?');

// Get messages
const messages = thread.toMessages();

// Use with invoke
const response = await invoke({
  messages: thread.toMessages(),
  model: 'llama-3.1-8b-instruct',
});

// Add response to thread
thread.addAssistant(response.text);

// Continue conversation
thread.addUser('What is 4+4?');
```
