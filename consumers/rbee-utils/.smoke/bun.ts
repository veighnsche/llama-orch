// Bun smoke test for @llama-orch/utils (WASM)
// Run with: bun ./.smoke/bun.ts
import utils from '../dist/index.js'

function assert(cond: any, msg?: string) {
	if (!cond) throw new Error(msg || 'assertion failed')
}

const r1 = utils.fs.readFile({ paths: ['README.md'], as_text: true, encoding: 'utf-8' })
assert(Array.isArray(r1.files), 'fs.readFile: files array')

const r2 = utils.fs.readFile({ paths: ['README.md'], as_text: false, encoding: null })
assert(r2.files[0].bytes && Array.isArray(r2.files[0].bytes), 'fs.readFile: bytes array present')

const pm = utils.prompt.message({ role: 'user', source: { Text: 'hello' }, dedent: false })
assert(pm.role === 'user' && pm.content === 'hello', 'prompt.message basic')

const pt = utils.prompt.thread({
	items: [
		{ role: 'system', source: { Text: 'a' }, dedent: false },
		{ role: 'user', source: { Lines: ['b', 'c'] }, dedent: true },
	],
})
assert(pt.messages.length === 2, 'prompt.thread size')

const m = utils.model.define({ model_id: 'm1', engine_id: null, pool_hint: 'pool-a' })
assert(m.model_id === 'm1', 'model.define id')

const p = utils.params.define({ temperature: 0.7, top_p: 1.0, max_tokens: 10, seed: null })
assert(typeof p.temperature === 'number', 'params.define out')

let threw = false
try {
	utils.llm.invoke({ messages: [{ role: 'user', content: 'hi' }], model: m, params: p })
} catch (e: any) {
	threw = /unimplemented: llm\.invoke requires SDK wiring/.test(String(e && e.message))
}
assert(threw, 'llm.invoke should throw typed unimplemented')

const result = { choices: [{ text: 'ok' }], usage: null }
const s = utils.orch.responseExtractor(result)
assert(typeof s === 'string', 'orch.responseExtractor string')

console.log('Bun smoke OK')
