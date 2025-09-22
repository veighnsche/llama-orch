// DRAFT smoke script (do not execute in this step)
// Adjust import path to point to the built addon output.
// Example for local run after build:
//   node --enable-source-maps -e "import('./index.js').then(m=>console.log(m))"  (placeholder)
// For napi-rs, the compiled addon is typically resolved via package entry.

import addon from '../npm/index.js'; // placeholder; adjust when packaging

async function main() {
  try {
    const res = addon.fs.readFile({ paths: ['README.md'], as_text: true, encoding: 'utf-8' });
    console.log('files:', res.files.length);
    console.log('first content length:', res.files[0]?.content?.length ?? 0);
  } catch (e) {
    console.error('smoke failed (expected in draft without packaging):', e?.message || e);
  }
}

main();
