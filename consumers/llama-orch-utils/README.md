# llama-orch-utils

M2 applets library for composing Blueprint pipelines in the llama-orch ecosystem. This crate hosts applets under `src/[namespace]/[applet]/` and is independent of SDK internals.

## WASI package (Node + Bun)

This directory also ships a WASI-based npm package that exposes the applets via a portable `.wasm` artifact and a minimal loader `index.js`.
The loader constructs a single default-exported API object from a Rust-provided manifest and routes all calls through a unified `invoke_json` dispatcher.

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
WASI_PREOPEN=/data/logs,/var/tmp node -e "import('./index.js').then(m => console.log(m.default.fs.read_file_json({ paths: ['README.md'], as_text: true, encoding: 'utf-8' })));"
```

### API surface (default-exported grouped object)

```ts
import utils from '@llama-orch/utils';

// fs
const r1 = utils.fs.read_file_json({ paths: ['README.md'], as_text: true, encoding: 'utf-8' });
const r2 = utils.fs.read_file_json({ paths: ['README.md'], as_text: false, encoding: null });

// prompt
const m1 = utils.prompt.message_json({ role: 'user', source: { Text: 'hello' }, dedent: false });
const t1 = utils.prompt.thread_json({ items: [
  { role: 'system', source: { Text: 'a' }, dedent: false },
  { role: 'user', source: { Lines: ['b', 'c'] }, dedent: true },
]});

// model / params
const mr = utils.model.define_json({ model_id: 'm1', engine_id: null, pool_hint: 'pool-a' });
const pr = utils.params.define_json({ temperature: 0.7, top_p: 1.0, max_tokens: 100, seed: null });

// llm (unimplemented in M2; throws typed error)
try { utils.llm.invoke_json({ messages: [{ role: 'user', content: 'hi' }], model: mr, params: pr }); }
catch (e) { /* expected */ }

// orch
const s = utils.orch.response_extractor_json({ choices: [{ text: 'ok' }], usage: null });
```

### Build

```bash
pnpm i
pnpm run build         # builds dist/llama_orch_utils.wasm + index.d.ts
pnpm run smoke:node    # Node smoke
pnpm run smoke:bun     # Bun smoke
```

If the WASM target is missing, install it (e.g., via `rustup target add wasm32-wasip1-threads`).
