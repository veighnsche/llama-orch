// Node smoke test for @llama-orch/utils (WASM)
import { fs, prompt, model, params, llm, orch } from '../index.js';

function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }

// fs.readFile string overload
const r1 = fs.readFile('README.md');
assert(Array.isArray(r1.files), 'fs.readFile: files array');
assert(r1.files.length >= 1, 'fs.readFile: at least one file');

// fs.readFile canonical shape
const r2 = fs.readFile({ paths: ['README.md'], as_text: false, encoding: null });
assert(r2.files[0].bytes && Array.isArray(r2.files[0].bytes), 'fs.readFile: bytes array present');

// prompt.message
const pm = prompt.message({ role: 'user', source: { Text: 'hello' }, dedent: false });
assert(pm.role === 'user' && pm.content === 'hello', 'prompt.message basic');

// prompt.thread
const pt = prompt.thread({ items: [
  { role: 'system', source: { Text: 'a' }, dedent: false },
  { role: 'user', source: { Lines: ['b', 'c'] }, dedent: true },
]});
assert(pt.messages.length === 2, 'prompt.thread size');

// model.define
const m = model.define('m1', null, 'pool-a');
assert(m.model_id === 'm1', 'model.define id');

// params.define
const p = params.define({ temperature: 0.7, top_p: 1.0, max_tokens: 10, seed: null });
assert(typeof p.temperature === 'number', 'params.define out');

// llm.invoke — should throw typed unimplemented
let threw = false;
try {
  llm.invoke({ messages: [{ role: 'user', content: 'hi' }], model: m, params: p });
} catch (e) {
  threw = /unimplemented: llm\.invoke requires SDK wiring/.test(String(e && e.message));
}
assert(threw, 'llm.invoke should throw typed unimplemented');

// orch.response_extractor
const result = { choices: [{ text: 'ok' }], usage: null };
const s = orch.response_extractor(result);
assert(typeof s === 'string', 'orch.response_extractor string');

console.log('Node smoke OK');
