# FileReader

**Read files from filesystem with async/streaming support**

```typescript
import { FileReader } from '@llama-orch/utils/fs';

// Read entire file
const content = await FileReader.read('input.txt');

// Stream large files
const stream = FileReader.stream('large-file.txt');
for await (const chunk of stream) {
  console.log(chunk);
}

// Read with encoding
const content = await FileReader.read('file.txt', { encoding: 'utf-8' });
```
