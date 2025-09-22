import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

// Load the napi-rs CommonJS loader after we rename it to index.cjs
const native = require('./index.cjs');

// Helper: drop keys whose value is strictly null
function dropNulls(obj, keys) {
  if (!obj || typeof obj !== 'object') return obj;
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, k) && obj[k] === null) {
      delete obj[k];
    }
  }
  return obj;
}

// Wrap llm.invoke to normalize nulls -> undefined (by deleting) for optional fields
function llmInvoke(input) {
  const cloned = input && typeof input === 'object' ? JSON.parse(JSON.stringify(input)) : input;
  if (cloned && cloned.model) {
    dropNulls(cloned.model, ['engineId', 'poolHint']);
  }
  if (cloned && cloned.params) {
    dropNulls(cloned.params, ['temperature', 'topP', 'maxTokens', 'seed']);
  }
  return native.llm.invoke(cloned);
}

// Construct the ESM facade surface
export const fs = { readFile: native.fs.readFile };
export const prompt = { message: native.prompt.message, thread: native.prompt.thread };
export const model = { define: native.model.define };
export const params = { define: native.params.define };
export const llm = { invoke: llmInvoke };
export const orch = { response_extractor: native.orch.responseExtractor, responseExtractor: native.orch.responseExtractor };
export const probe = native.probe;

const defaultExport = { fs, prompt, model, params, llm, orch, probe };
export default defaultExport;
