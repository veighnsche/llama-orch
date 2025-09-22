#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { promises as fsp } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');

const TARGET = 'wasm32-wasip1-threads';
const CRATE = 'llama-orch-utils';
const OUT_NAME = 'llama_orch_utils.wasm';
const OUT_DIR = resolve(root, 'dist');
const OUT_PATH = resolve(OUT_DIR, OUT_NAME);
const BUILD_ARTIFACT = resolve(root, `../../target/${TARGET}/release/${OUT_NAME}`);

function run(cmd, args, opts = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(cmd, args, { stdio: 'inherit', cwd: root, ...opts });
    child.on('exit', (code) => {
      if (code === 0) resolvePromise(); else reject(new Error(`${cmd} exited with ${code}`));
    });
  });
}

(async () => {
  try {
    await run('cargo', ['build', '--release', '--target', TARGET, '-p', CRATE]);
    await fsp.mkdir(OUT_DIR, { recursive: true });
    await fsp.copyFile(BUILD_ARTIFACT, OUT_PATH);
    console.log(`Copied ${BUILD_ARTIFACT} -> ${OUT_PATH}`);
  } catch (err) {
    console.error('build-wasm failed:', err.message || err);
    process.exit(1);
  }
})();
