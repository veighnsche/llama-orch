// WASI loader for @llama-orch/utils (WASM)
// - Instantiates wasm32-wasip1-threads build of the core crate
// - Exposes grouped API { fs, prompt, model, params, llm, orch }
// - Provides fs.readFile overload: (path: string)

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { promises as fsp } from 'node:fs';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = resolve(here, 'dist/llama_orch_utils.wasm');

let _exports; // wasm exports
let _memory;  // WebAssembly.Memory

async function init() {
  if (_exports) return;

  // Resolve WASI implementation (Node >= 20 preferred). Fall back to alternate specifiers.
  let WASIClass;
  try {
    ({ WASI: WASIClass } = await import('node:wasi'));
  } catch {
    try {
      ({ WASI: WASIClass } = await import('wasi'));
    } catch {
      try {
        ({ WASI: WASIClass } = await import('bun:ffi'));
      } catch {
        throw new Error('WASI implementation not available. Require Node >= 20 (node:wasi) or Bun with WASI support.');
      }
    }
  }

  // Preopens: default to cwd; optionally add paths from WASI_PREOPEN (comma-separated absolute paths)
  const preopens = { '/': process.cwd() };
  const envPreopen = process.env.WASI_PREOPEN;
  if (envPreopen && envPreopen.trim().length > 0) {
    const parts = envPreopen.split(',').map(s => s.trim()).filter(Boolean);
    parts.forEach((p, i) => { preopens[`/mnt${i}`] = p; });
  }

  const wasi = new WASIClass({
    version: 'preview1',
    preopens,
    args: process.argv,
    env: process.env,
  });

  const bytes = await fsp.readFile(wasmPath);
  const module = await WebAssembly.compile(bytes);

  // Get WASI imports from API
  const wasiImports = (typeof wasi.getImportObject === 'function')
    ? wasi.getImportObject()
    : (wasi && wasi.wasiImport ? { wasi_snapshot_preview1: wasi.wasiImport } : {});

  // Helper to attempt instantiation with a specific memory limit
  async function instantiateWith(maxPages) {
    const mem = new WebAssembly.Memory({ initial: maxPages, maximum: maxPages, shared: true });
    const importObject = { 
      ...wasiImports,
      env: { ...(wasiImports.env || {}), memory: mem },
    };
    // Some WASI implementations (e.g., Bun's) expect `this.memory` to be present before any import thunk runs.
    try { if (wasi && typeof wasi === 'object') { wasi.memory = mem; } } catch {}
    const inst = await WebAssembly.instantiate(module, importObject);
    return { inst, mem };
  }

  let instance, memory;
  try {
    // First try a generous default; if too large, we'll parse the module's declared max from the error and retry
    ({ inst: instance, mem: memory } = await instantiateWith(256));
  } catch (e) {
    const msg = String(e && e.message || e);
    // Try to parse a numeric maximum (Node usually reports it)
    const m = msg.match(/module's declared maximum\s+(\d+)/i);
    if (m) {
      const moduleMax = Math.max(1, Math.min(65536, parseInt(m[1], 10) || 17));
      ({ inst: instance, mem: memory } = await instantiateWith(moduleMax));
    } else {
      // Bun doesn't report the number; try a descending set of candidates until success
      const candidates = [256, 128, 64, 32, 17, 16, 8, 4, 2, 1];
      let lastErr = e;
      for (const pages of candidates) {
        try {
          ({ inst: instance, mem: memory } = await instantiateWith(pages));
          lastErr = null;
          break;
        } catch (err) {
          lastErr = err;
        }
      }
      if (!instance) throw lastErr || e;
    }
  }

  if (typeof wasi.initialize === 'function') wasi.initialize(instance);

  _exports = instance.exports;
  _memory = (instance.exports && instance.exports.memory) ? instance.exports.memory : memory;
}

function ensureInitSyncGuard() {
  if (!_exports) throw new Error('WASM not initialized yet. Ensure module evaluation completed.');
}

function encode(obj) {
  const json = JSON.stringify(obj);
  const buf = Buffer.from(json);
  const ptr = Number(_exports.alloc(buf.length) >>> 0);
  new Uint8Array(_memory.buffer, ptr, buf.length).set(buf);
  return { ptr, len: buf.length };
}

function decode(ret64) {
  // ret64 is a BigInt encoding: (len << 32) | ptr
  const v = BigInt(ret64);
  const ptr = Number(v & 0xFFFFFFFFn) >>> 0;
  const len = Number((v >> 32n) & 0xFFFFFFFFn) >>> 0;
  const view = new Uint8Array(_memory.buffer, ptr, len);
  const out = Buffer.from(view).toString('utf8');
  // free buffer (capacity == len by construction)
  if (typeof _exports.dealloc === 'function') {
    _exports.dealloc(ptr, len);
  } else if (typeof _exports.free === 'function') {
    // fallback for older builds
    _exports.free(ptr, len);
  }
  const parsed = JSON.parse(out);
  if (parsed && typeof parsed === 'object' && parsed.error) {
    const msg = typeof parsed.message === 'string' ? parsed.message : JSON.stringify(parsed.message);
    throw new Error(msg);
  }
  return parsed;
}

function callJson(funcName, inputObj) {
  ensureInitSyncGuard();
  const { ptr, len } = encode(inputObj);
  const ret = _exports[funcName](ptr, len);
  return decode(ret);
}

await init();

export const fs = {
  readFile(inputOrPath) {
    if (typeof inputOrPath === 'string') {
      return callJson('fs_read_file_json', { paths: [inputOrPath], as_text: true, encoding: 'utf-8' });
    }
    return callJson('fs_read_file_json', inputOrPath);
  },
  writeFile(input) {
    return callJson('fs_write_file_json', input);
  },
};

export const prompt = {
  message(input) { return callJson('prompt_message_json', input); },
  thread(input) { return callJson('prompt_thread_json', input); },
};

export const model = {
  define(model_id, engine_id, pool_hint) {
    return callJson('model_define_json', { model_id, engine_id: engine_id ?? null, pool_hint: pool_hint ?? null });
  },
};

export const params = {
  define(p) { return callJson('params_define_json', p); },
};

export const llm = {
  invoke(input) { return callJson('llm_invoke_json', input); },
};

export const orch = {
  responseExtractor(result) { return callJson('orch_response_extractor_json', result); },
};

const _default = { fs, prompt, model, params, llm, orch };
export default _default;
