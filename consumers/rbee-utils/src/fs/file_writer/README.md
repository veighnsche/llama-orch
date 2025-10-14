# FileWriter

**Write files to filesystem with async/streaming support**

```typescript
import { FileWriter } from '@llama-orch/utils/fs';

// Write entire file
await FileWriter.write('output.txt', 'Hello, world!');

// Append to file
await FileWriter.append('log.txt', 'New log entry\n');

// Write with encoding
await FileWriter.write('file.txt', content, { encoding: 'utf-8' });

// Stream write
const writer = FileWriter.createStream('output.txt');
writer.write('chunk 1');
writer.write('chunk 2');
await writer.close();
```
