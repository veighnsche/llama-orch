// Node smoke test for @llama-orch/utils (WASM)
import utils from '../dist/index.js'

function assert(cond, msg) {
	if (!cond) throw new Error(msg || 'assertion failed')
}

// fs.read_file_json canonical request
const r1 = utils.fs.read_file_json({ paths: ['README.md'], as_text: true, encoding: 'utf-8' })
assert(Array.isArray(r1.files), 'fs.readFile: files array')
assert(r1.files.length >= 1, 'fs.readFile: at least one file')

// fs.read_file_json binary request
const r2 = utils.fs.read_file_json({ paths: ['README.md'], as_text: false, encoding: null })
assert(r2.files[0].bytes && Array.isArray(r2.files[0].bytes), 'fs.readFile: bytes array present')

// prompt.message_json
const pm = utils.prompt.message_json({ role: 'user', source: { Text: 'hello' }, dedent: false })
assert(pm.role === 'user' && pm.content === 'hello', 'prompt.message basic')

// prompt.thread_json
const pt = utils.prompt.thread_json({
	items: [
		{ role: 'system', source: { Text: 'a' }, dedent: false },
		{ role: 'user', source: { Lines: ['b', 'c'] }, dedent: true },
	],
})
assert(pt.messages.length === 2, 'prompt.thread size')

// model.define_json
const m = utils.model.define_json({ model_id: 'm1', engine_id: null, pool_hint: 'pool-a' })
assert(m.model_id === 'm1', 'model.define id')

// params.define_json
const p = utils.params.define_json({ temperature: 0.7, top_p: 1.0, max_tokens: 10, seed: null })
assert(typeof p.temperature === 'number', 'params.define out')

// llm.invoke — should throw typed unimplemented
let threw = false
try {
	utils.llm.invoke_json({ messages: [{ role: 'user', content: 'hi' }], model: m, params: p })
} catch (e) {
	threw = /unimplemented: llm\.invoke requires SDK wiring/.test(String(e && e.message))
}
assert(threw, 'llm.invoke should throw typed unimplemented')

// orch.response_extractor_json
const result = { choices: [{ text: 'ok' }], usage: null }
const s = utils.orch.response_extractor_json(result)
assert(typeof s === 'string', 'orch.response_extractor string')

console.log('Node smoke OK')
