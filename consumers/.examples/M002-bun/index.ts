import utils from '@llama-orch/utils';

// Use a path relative to cwd. The WASI loader preopens cwd as '/'.
const seedPath = './.llama-orch/seed.md';

// Call grouped default export with canonical request shape.
const res = utils.fs.readFile({ paths: [seedPath], as_text: true, encoding: 'utf-8' });
const first = res.files[0];
if (!first || first.content == null) {
  throw new Error(`Failed to read ${seedPath}`);
}
console.log(first.content);
