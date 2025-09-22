// Node smoke test for @llama-orch/utils (WASM)
import {
  fs_read_file_json,
  fs_write_file_json,
  prompt_message_json,
  prompt_thread_json,
  model_define_json,
  params_define_json,
  llm_invoke_json,
  orch_response_extractor_json,
} from '../index.js';

function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }

// fs_read_file_json canonical request
const r1 = fs_read_file_json({ paths: ['README.md'], as_text: true, encoding: 'utf-8' });
assert(Array.isArray(r1.files), 'fs.readFile: files array');
assert(r1.files.length >= 1, 'fs.readFile: at least one file');

// fs_read_file_json binary request
const r2 = fs_read_file_json({ paths: ['README.md'], as_text: false, encoding: null });
assert(r2.files[0].bytes && Array.isArray(r2.files[0].bytes), 'fs.readFile: bytes array present');

// prompt_message_json
const pm = prompt_message_json({ role: 'user', source: { Text: 'hello' }, dedent: false });
assert(pm.role === 'user' && pm.content === 'hello', 'prompt.message basic');

// prompt_thread_json
const pt = prompt_thread_json({ items: [
  { role: 'system', source: { Text: 'a' }, dedent: false },
  { role: 'user', source: { Lines: ['b', 'c'] }, dedent: true },
]});
assert(pt.messages.length === 2, 'prompt.thread size');

// model_define_json
const m = model_define_json({ model_id: 'm1', engine_id: null, pool_hint: 'pool-a' });
assert(m.model_id === 'm1', 'model.define id');

// params_define_json
const p = params_define_json({ temperature: 0.7, top_p: 1.0, max_tokens: 10, seed: null });
assert(typeof p.temperature === 'number', 'params.define out');

// llm.invoke — should throw typed unimplemented
let threw = false;
try {
  llm_invoke_json({ messages: [{ role: 'user', content: 'hi' }], model: m, params: p });
} catch (e) {
  threw = /unimplemented: llm\.invoke requires SDK wiring/.test(String(e && e.message));
}
assert(threw, 'llm.invoke should throw typed unimplemented');

// orch_response_extractor_json
const result = { choices: [{ text: 'ok' }], usage: null };
const s = orch_response_extractor_json(result);
assert(typeof s === 'string', 'orch.response_extractor string');

console.log('Node smoke OK');
