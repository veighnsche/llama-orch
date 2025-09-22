// Smoke: fs.readFile happy path (ESM)
import { promises as fsp } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fs } from '../esm.js';

function fail(msg) { console.error(msg); process.exit(1); }
function ok() { console.log('OK'); process.exit(0); }

try {
  const p = join(tmpdir(), `llorch-smoke-${Date.now()}.txt`);
  await fsp.writeFile(p, 'hello\nworld', 'utf-8');
  const res = fs.readFile({ paths: [p], asText: true, encoding: 'utf-8' });
  if (!res || !res.files || res.files.length !== 1) fail('expected one file');
  if (!res.files[0].path.endsWith('.txt')) fail('unexpected path');
  if (res.files[0].content !== 'hello\nworld') fail(`unexpected content: ${res.files[0].content}`);
  if (res.files[0].bytes != null) fail('expected bytes to be null/undefined for as_text');
  ok();
} catch (e) {
  fail(e?.message || String(e));
}
