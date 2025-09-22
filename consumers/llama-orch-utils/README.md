# llama-orch-utils

M2 applets library for composing Blueprint pipelines in the llama-orch ecosystem. This crate hosts applets under `src/[namespace]/[applet]/` and is independent of SDK internals.

## WASI package (Node + Bun)

This directory also ships a WASI-based npm package that exposes the applets via a portable `.wasm` artifact and a minimal loader `index.js`.

- Runtime requirements:
  - Node >= 20 (built-in `node:wasi`) or Bun with WASI support.
  - The `.wasm` is built for `wasm32-wasip1-threads`.
- Build requirements:
  - Rust target `wasm32-wasip1-threads` installed.
  - `pnpm` (or `npm`) to run scripts.

### Preopened directories

By default, the loader preopens the current working directory (cwd) as `/`. You can add additional preopens by setting `WASI_PREOPEN` to a comma-separated list of absolute host paths, which will be mounted at `/mnt0`, `/mnt1`, ... inside WASI.

Example:

```bash
WASI_PREOPEN=/data/logs,/var/tmp node -e "import('./index.js').then(m => console.log(m.fs.readFile('README.md')));"
```

### API surface (flat 1:1 function exports)

```ts
import {
  fs_read_file_json,
  fs_write_file_json,
  prompt_message_json,
  prompt_thread_json,
  model_define_json,
  params_define_json,
  llm_invoke_json,
  orch_response_extractor_json,
} from '@llama-orch/utils';

// fs
const r1 = fs_read_file_json({ paths: ['README.md'], as_text: true, encoding: 'utf-8' });
const r2 = fs_read_file_json({ paths: ['README.md'], as_text: false, encoding: null });

// prompt
const m1 = prompt_message_json({ role: 'user', source: { Text: 'hello' }, dedent: false });
const t1 = prompt_thread_json({ items: [
  { role: 'system', source: { Text: 'a' }, dedent: false },
  { role: 'user', source: { Lines: ['b', 'c'] }, dedent: true },
]});

// model / params
const mr = model_define_json({ model_id: 'm1', engine_id: null, pool_hint: 'pool-a' });
const pr = params_define_json({ temperature: 0.7, top_p: 1.0, max_tokens: 100, seed: null });

// llm (unimplemented in M2; throws typed error)
try { llm_invoke_json({ messages: [{ role: 'user', content: 'hi' }], model: mr, params: pr }); }
catch (e) { /* expected */ }

// orch
const s = orch_response_extractor_json({ choices: [{ text: 'ok' }], usage: null });
```

### Build

```bash
pnpm i
pnpm run build         # builds dist/llama_orch_utils.wasm + index.d.ts
pnpm run smoke:node    # Node smoke
pnpm run smoke:bun     # Bun smoke
```

If the WASM target is missing, install it (e.g., via `rustup target add wasm32-wasip1-threads`).
