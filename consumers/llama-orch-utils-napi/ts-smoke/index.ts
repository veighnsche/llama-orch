/// <reference path="./shims.d.ts" />
import { fs, prompt, model, params, llm, orch } from '../esm.js';

// Type-only smoke: no runtime execute is needed; tsc should validate signatures.
const r = fs.readFile({ paths: ['x'], asText: true, encoding: 'utf-8' });
const m = prompt.message({ role: 'user', source: { kind: 'Text', text: 'hi' }, dedent: false });
const mr = model.define('gpt-foo', null, null);
const p = params.define({ temperature: 0.7, topP: 1, maxTokens: 100, seed: null });
const s = orch.response_extractor({ choices: [{ text: 'ok' }], usage: { promptTokens: 1, completionTokens: 2 } });
try { llm.invoke({ messages: [], model: { modelId: 'x', engineId: null, poolHint: null }, params: {} as any }); } catch {}
