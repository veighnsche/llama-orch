# Message

**Construct individual chat messages**

```typescript
import { Message } from '@llama-orch/utils/prompt';

// Create messages
const system = Message.system('You are a helpful assistant');
const user = Message.user('Hello, world!');
const assistant = Message.assistant('Hi! How can I help?');

// With metadata
const message = Message.user('Question', {
  name: 'User',
  timestamp: Date.now(),
});

// Convert to API format
const apiMessage = message.toAPI();
// { role: 'user', content: 'Question' }
```
