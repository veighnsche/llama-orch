// Smoke: prompt.message happy path (ESM)
import { prompt } from '../esm.js';

function fail(msg) { console.error(msg); process.exit(1); }
function ok() { console.log('OK'); process.exit(0); }

try {
  const out = prompt.message({
    role: 'user',
    source: { kind: 'Text', text: 'hi' },
    dedent: false,
  });
  if (!out || out.role !== 'user' || out.content !== 'hi') {
    fail(`unexpected output: ${JSON.stringify(out)}`);
  }
  ok();
} catch (e) {
  fail(e?.message || String(e));
}
