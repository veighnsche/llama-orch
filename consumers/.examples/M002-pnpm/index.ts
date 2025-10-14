import utils from '@rbee/utils';

// Use a path relative to cwd. The WASI loader preopens cwd as '/'.
const seedPath = './.rbee/seed.md';

// fs.readFile (text)
const rText = utils.fs.readFile({ paths: [seedPath], as_text: true, encoding: 'utf-8' });
const firstText = rText.files[0];
if (!firstText || firstText.content == null) {
  throw new Error(`Failed to read ${seedPath}`);
}
console.log('[fs.readFile:text] chars:', firstText.content.length);

// fs.readFile (binary)
const rBin = utils.fs.readFile({ paths: [seedPath], as_text: false, encoding: null });
const firstBin = rBin.files[0];
console.log('[fs.readFile:bin] bytes:', firstBin && firstBin.bytes ? firstBin.bytes.length : 0);

// fs.writeFile
const out = utils.fs.writeFile({ path: './.rbee/out.txt', text: 'hello from utils\n', create_dirs: true });
console.log('[fs.writeFile] wrote:', out.bytes_written, 'bytes to', out.path);

// prompt.message
const pm = utils.prompt.message({ role: 'user', source: { Text: 'hello' }, dedent: false });
console.log('[prompt.message] role:', pm.role, 'content:', pm.content);

// prompt.thread
const pt = utils.prompt.thread({ items: [
  { role: 'system', source: { Text: 'You are a helpful assistant.' }, dedent: false },
  { role: 'user', source: { Lines: ['How are you?', 'Answer briefly.'] }, dedent: true },
]});
console.log('[prompt.thread] messages:', pt.messages.length);

// model.define
const model = utils.model.define({ model_id: 'm1', engine_id: null, pool_hint: 'pool-a' });
console.log('[model.define] model_id:', model.model_id);

// params.define
const params = utils.params.define({ temperature: 0.7, top_p: 1.0, max_tokens: 16, seed: null });
console.log('[params.define] temperature:', params.temperature);

// llm.invoke (expected to throw typed 'unimplemented')
let threw = false;
try {
  utils.llm.invoke({ messages: [{ role: 'user', content: 'hi' }], model, params });
} catch (e) {
  threw = true;
  console.log('[llm.invoke] threw as expected:', String((e as any)?.message || e));
}
if (!threw) {
  throw new Error('llm.invoke should throw typed unimplemented in M2');
}

// orch.responseExtractor
const extracted = utils.orch.responseExtractor({ choices: [{ text: 'ok' }], usage: null });
console.log('[orch.responseExtractor] extracted:', extracted);
