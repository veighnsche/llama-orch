# Response Extractors

**Extract structured data from LLM responses**

```typescript
import { extractJson, extractCode, extractMarkdown } from '@llama-orch/utils/orch';

// Extract JSON
const response = 'Here is the data: {"name": "John", "age": 30}';
const data = extractJson(response);
console.log(data.name); // "John"

// Extract code blocks
const response = '```typescript\nconst x = 42;\n```';
const code = extractCode(response, 'typescript');
console.log(code); // "const x = 42;"

// Extract all code blocks
const blocks = extractCode(response); // Array of all code blocks

// Extract markdown sections
const sections = extractMarkdown(response, '## Section');
```
