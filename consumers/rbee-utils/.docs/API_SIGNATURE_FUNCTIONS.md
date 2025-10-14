# Per-Function API Signatures (Current vs Proposed)

Edit the Proposed code fences under each function with the exact signature(s) you want users to call.
I will wire the generator to provide those ergonomic overloads while keeping Rust as the canonical
request/response shapes internally.

Notation: all shapes are inlined here (no type aliases) so you can tweak names and fields directly.

---

## fs.readFile

- Current usage (from example):
```ts
utils.fs.readFile({ paths: [seedPath], as_text: true, encoding: 'utf-8' })
utils.fs.readFile({ paths: [seedPath], as_text: false, encoding: null })
```

- Current canonical declaration (inlined):
```ts
function readFile(input: {
  paths: string[];
  as_text: boolean;
  encoding: string | null; // only 'utf-8' supported in M2; null for binary
}): {
  files: Array<{
    path: string;
    content: string | null;
    bytes: number[] | null;
  }>
}
```

- Ergonomic variants you can choose from:
```ts
// Single path, text by default
function readFile(path: string, opts?: { asText?: boolean; encoding?: 'utf-8' | null }): {
  files: Array<{ path: string; content: string | null; bytes: number[] | null }>
}

// Multiple paths
function readFile(paths: string[], opts?: { asText?: boolean; encoding?: 'utf-8' | null }): {
  files: Array<{ path: string; content: string | null; bytes: number[] | null }>
}
```

- Proposed (edit me):
```ts
// Replace with exactly the signatures you want to ship for fs.readFile
function readFile(input: {
  paths: string[];
  as_text: boolean;
  encoding: string | null;
}): {
  files: Array<{ path: string; content: string | null; bytes: number[] | null }>
}
```

---

## fs.writeFile

- Current usage (from example):
```ts
utils.fs.writeFile({ path: './.llama-orch/out.txt', text: 'hello', create_dirs: true })
```

- Current canonical declaration (inlined):
```ts
function writeFile(input: { path: string; text: string; create_dirs: boolean }): {
  path: string;
  bytes_written: number;
}
```

- Ergonomic variants you can choose from:
```ts
function writeFile(path: string, text: string, opts?: { createDirs?: boolean }): {
  path: string;
  bytes_written: number;
}
```

- Proposed (edit me):
```ts
function writeFile(input: { path: string; text: string; create_dirs: boolean }): {
  path: string;
  bytes_written: number;
}
```

---

## prompt.message

- Current usage (from example):
```ts
utils.prompt.message({ role: 'user', source: { Text: 'hello' }, dedent: false })
```

- Current canonical declaration (inlined):
```ts
function message(input: {
  role: string;
  source: ({ Text: string } | { Lines: string[] } | { File: string });
  dedent: boolean;
}): {
  role: string;
  content: string;
}
```

- Ergonomic variants you can choose from:
```ts
function message(role: 'system' | 'user' | 'assistant', content: string, opts?: { dedent?: boolean }): {
  role: string;
  content: string;
}
```

- Proposed (edit me):
```ts
function message(input: {
  role: string;
  source: ({ Text: string } | { Lines: string[] } | { File: string });
  dedent: boolean;
}): { role: string; content: string }
```

---

## prompt.thread

- Current usage (from example):
```ts
utils.prompt.thread({ items: [
  { role: 'system', source: { Text: 'You are a helpful assistant.' }, dedent: false },
  { role: 'user', source: { Lines: ['How are you?', 'Answer briefly.'] }, dedent: true },
]})
```

- Current canonical declaration (inlined):
```ts
function thread(input: {
  items: Array<{
    role: string;
    source: ({ Text: string } | { Lines: string[] } | { File: string });
    dedent: boolean;
  }>
}): {
  messages: Array<{ role: string; content: string }>
}
```

- Ergonomic variants you can choose from:
```ts
function thread(items: Array<{ role: string; content: string } | string>): {
  messages: Array<{ role: string; content: string }>
}
```

- Proposed (edit me):
```ts
function thread(input: {
  items: Array<{ role: string; source: ({ Text: string } | { Lines: string[] } | { File: string }); dedent: boolean }>
}): { messages: Array<{ role: string; content: string }> }
```

---

## model.define

- Current usage (from example):
```ts
utils.model.define({ model_id: 'm1', engine_id: null, pool_hint: 'pool-a' })
```

- Current canonical declaration (inlined):
```ts
function define(input: {
  model_id: string;
  engine_id: string | null;
  pool_hint: string | null;
}): {
  model_id: string;
  engine_id: string | null;
  pool_hint: string | null;
}
```

- Ergonomic variants you can choose from:
```ts
function define(modelId: string, engineId?: string | null, poolHint?: string | null): {
  model_id: string;
  engine_id: string | null;
  pool_hint: string | null;
}
```

- Proposed (edit me):
```ts
function define(input: { model_id: string; engine_id: string | null; pool_hint: string | null }): {
  model_id: string; engine_id: string | null; pool_hint: string | null
}
```

---

## params.define

- Current usage (from example):
```ts
utils.params.define({ temperature: 0.7, top_p: 1.0, max_tokens: 16, seed: null })
```

- Current canonical declaration (inlined):
```ts
function define(p: {
  temperature?: number | null;
  top_p?: number | null;
  max_tokens?: number | null;
  seed?: number | null;
}): {
  temperature: number | null;
  top_p: number | null;
  max_tokens: number | null;
  seed: number | null;
}
```

- Ergonomic variants you can choose from:
```ts
function define(partial: Partial<{ temperature: number; top_p: number; max_tokens: number; seed: number | null }>): {
  temperature: number | null;
  top_p: number | null;
  max_tokens: number | null;
  seed: number | null;
}
```

- Proposed (edit me):
```ts
function define(p: { temperature?: number | null; top_p?: number | null; max_tokens?: number | null; seed?: number | null }): {
  temperature: number | null; top_p: number | null; max_tokens: number | null; seed: number | null
}
```

---

## llm.invoke

- Current usage (from example):
```ts
utils.llm.invoke({ messages: [{ role: 'user', content: 'hi' }], model, params })
```

- Current canonical declaration (inlined):
```ts
function invoke(input: {
  messages: Array<{ role: string; content: string }>;
  model: { model_id: string; engine_id: string | null; pool_hint: string | null };
  params: { temperature?: number | null; top_p?: number | null; max_tokens?: number | null; seed?: number | null };
}): {
  result: {
    choices: Array<{ text: string }>;
    usage: { prompt_tokens?: number | null; completion_tokens?: number | null } | null;
  }
}
```

- Ergonomic variants you can choose from:
```ts
function invoke(messages: Array<{ role: string; content: string }>,
                model: { model_id: string; engine_id: string | null; pool_hint: string | null },
                params: Partial<{ temperature: number; top_p: number; max_tokens: number; seed: number | null }>): {
  result: { choices: Array<{ text: string }>; usage: { prompt_tokens?: number | null; completion_tokens?: number | null } | null }
}
```

- Proposed (edit me):
```ts
function invoke(input: {
  messages: Array<{ role: string; content: string }>;
  model: { model_id: string; engine_id: string | null; pool_hint: string | null };
  params: { temperature?: number | null; top_p?: number | null; max_tokens?: number | null; seed?: number | null };
}): { result: { choices: Array<{ text: string }>; usage: { prompt_tokens?: number | null; completion_tokens?: number | null } | null } }
```

---

## orch.responseExtractor

- Current usage (from example):
```ts
utils.orch.responseExtractor({ choices: [{ text: 'ok' }], usage: null })
```

- Current canonical declaration (inlined):
```ts
function responseExtractor(result: {
  choices: Array<{ text: string }>;
  usage: { prompt_tokens?: number | null; completion_tokens?: number | null } | null;
}): string
```

- Ergonomic variants you can choose from:
```ts
function extractText(result: { choices: Array<{ text: string }>; usage: { prompt_tokens?: number | null; completion_tokens?: number | null } | null }): string
```

- Proposed (edit me):
```ts
function responseExtractor(result: { choices: Array<{ text: string }>; usage: { prompt_tokens?: number | null; completion_tokens?: number | null } | null }): string
```
