// DRAFT smoke script (do not execute in this step)
// Example shape for Source:
// { kind: "Text", text: "hi" }
// { kind: "Lines", lines: ["a", "b"] }
// { kind: "File", path: "./prompt.txt" }

import addon from '../npm/index.js'; // placeholder path; adjust after packaging

function demo() {
  try {
    const msg = addon.prompt.message({ role: 'user', source: { kind: 'Text', text: 'hi' }, dedent: false });
    console.log('content:', msg.content);
  } catch (e) {
    console.error('prompt.message smoke failed (expected until packaged):', e?.message || e);
  }
}

demo();
