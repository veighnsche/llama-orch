// DRAFT smoke script for llm.invoke (Path B)
// This script loads the built addon via the CLI-generated loader (index.js)
// and asserts the exported llm.invoke throws the exact UNIMPLEMENTED message.

import * as addon from '../esm.js';

const invoke =
  addon?.llm?.invoke ??
  addon?.invoke ??
  addon?.llm_invoke ??
  addon?.default?.llm?.invoke ??
  addon?.default?.invoke ??
  addon?.default?.llm_invoke;

if (typeof invoke !== 'function') {
  console.error('No `invoke` export found.', {
    keys: Object.keys(addon),
    defaultKeys: addon?.default && Object.keys(addon.default),
  });
  process.exit(1);
}

function fail(msg) {
  console.error(msg);
  process.exit(1);
}

function ok(msg) {
  console.log(msg);
  process.exit(0);
}

try {
  const input = {
    messages: [],
    model: { modelId: 'dummy', engineId: null, poolHint: null },
    params: { temperature: null, topP: null, maxTokens: null, seed: null },
  };
  // Expect this to throw an Error with the exact message
  // "unimplemented: llm.invoke requires SDK wiring"
  const _ = invoke(input);
  fail('Expected llm.invoke to throw, but it returned successfully');
} catch (e) {
  const message = e?.message ?? String(e);
  const expected = 'unimplemented: llm.invoke requires SDK wiring';
  if (message === expected) {
    ok('OK');
  } else {
    fail(`Unexpected error message: ${message}`);
  }
}
