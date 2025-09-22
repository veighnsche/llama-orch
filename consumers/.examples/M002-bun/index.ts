import { fs_read_file_json } from '@llama-orch/utils';

// Use a path relative to cwd. The WASI loader preopens cwd as '/'.
const seedPath = './.llama-orch/seed.md';

// Call flat function export with canonical request shape.
const res = fs_read_file_json({ paths: [seedPath], as_text: true, encoding: 'utf-8' });
if (!res.files.length || res.files[0].content == null) {
  throw new Error(`Failed to read ${seedPath}`);
}
console.log(res.files[0].content);
