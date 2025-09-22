// Postbuild guard: ensure napi-rs generated index.d.ts exists next to index.cjs
import { existsSync } from 'node:fs';

const required = [
  'index.cjs',
  'index.d.ts',
];

const missing = required.filter((p) => !existsSync(new URL(`../${p}`, import.meta.url)));

if (missing.length > 0) {
  console.error('[postbuild:guard] Missing required build artifacts:', missing.join(', '));
  console.error('Expected napi-rs to generate index.d.ts alongside index.cjs.');
  process.exit(1);
}

process.exit(0);
