import { fs } from '@llama-orch/utils';

// Use a path relative to cwd. The WASI loader preopens cwd as '/'.
const seedPath = './.llama-orch/seed.md';

// fs.readFile supports a string overload that reads as text (utf-8) by default.
const res = fs.readFile(seedPath);
if (!res.files.length || res.files[0].content == null) {
  throw new Error(`Failed to read ${seedPath}`);
}
console.log(res.files[0].content);
