// WASI loader for @llama-orch/utils (WASM)
// - Instantiates wasm32-wasip1-threads build of the core crate
// - Exposes grouped API { fs, prompt, model, params, llm, orch }
// - Provides fs.readFile overload: (path: string)

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { promises as fsp } from 'node:fs';

const here = dirname(fileURLToPath(import.meta.url));
// When compiled, this file lives in dist/, alongside the wasm artifact.
const wasmPath = resolve(here, 'llama_orch_utils.wasm');

let _exports: any; // wasm exports
let _memory: WebAssembly.Memory;  // WebAssembly.Memory

async function init() {
  if (_exports) return;

  // Resolve WASI implementation (Node >= 20 preferred). Fall back to alternate specifiers.
  let WASIClass: any;
  try {
    ({ WASI: WASIClass } = await import('node:wasi' as any));
  } catch {
    try {
      ({ WASI: WASIClass } = await import('wasi' as any));
    } catch {
      try {
        ({ WASI: WASIClass } = await import('bun:ffi' as any));
      } catch {
        throw new Error('WASI implementation not available. Require Node >= 20 (node:wasi) or Bun with WASI support.');
      }
    }
  }

  // Preopens: default to cwd; optionally add paths from WASI_PREOPEN (comma-separated absolute paths)
  const preopens: Record<string, string> = { '/': process.cwd() };
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
  const wasiImports: any = (typeof wasi.getImportObject === 'function')
    ? wasi.getImportObject()
    : (wasi && wasi.wasiImport ? { wasi_snapshot_preview1: wasi.wasiImport } : {});

  // Helper to attempt instantiation with a specific memory limit
  async function instantiateWith(maxPages: number) {
    const mem = new WebAssembly.Memory({ initial: maxPages, maximum: maxPages, shared: true });
    const importObject: any = {
      ...wasiImports,
      env: { ...(wasiImports.env || {}), memory: mem },
    };
    // Some WASI implementations (e.g., Bun's) expect `this.memory` to be present before any import thunk runs.
    try { if (wasi && typeof wasi === 'object') { wasi.memory = mem; } } catch {}
    const inst = await WebAssembly.instantiate(module, importObject);
    return { inst, mem };
  }

  let instance: WebAssembly.Instance | undefined, memory: WebAssembly.Memory | undefined;
  try {
    // First try a generous default; if too large, we'll parse the module's declared max from the error and retry
    ({ inst: instance as any, mem: memory } = await instantiateWith(256));
  } catch (e: any) {
    const msg = String(e && e.message || e);
    // Try to parse a numeric maximum (Node usually reports it)
    const m = msg.match(/module's declared maximum\s+(\d+)/i);
    if (m) {
      const moduleMax = Math.max(1, Math.min(65536, parseInt(m[1], 10) || 17));
      ({ inst: instance as any, mem: memory } = await instantiateWith(moduleMax));
    } else {
      // Bun doesn't report the number; try a descending set of candidates until success
      const candidates = [256, 128, 64, 32, 17, 16, 8, 4, 2, 1];
      let lastErr: any = e;
      for (const pages of candidates) {
        try {
          ({ inst: instance as any, mem: memory } = await instantiateWith(pages));
          lastErr = null;
          break;
        } catch (err) {
          lastErr = err;
        }
      }
      if (!instance) throw lastErr || e;
    }
  }

  if (typeof (wasi as any).initialize === 'function') (wasi as any).initialize(instance);

  _exports = (instance as any).exports;
  _memory = ((instance as any).exports && (instance as any).exports.memory) ? (instance as any).exports.memory : (memory as WebAssembly.Memory);
}

function ensureInitSyncGuard() {
  if (!_exports) throw new Error('WASM not initialized yet. Ensure module evaluation completed.');
}

function encode(obj: any) {
  const json = JSON.stringify(obj);
  const buf = Buffer.from(json);
  const ptr = Number(_exports.alloc(buf.length) >>> 0);
  new Uint8Array(_memory.buffer, ptr, buf.length).set(buf);
  return { ptr, len: buf.length };
}

function decode(ret64: bigint) {
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

function callJson(funcName: string, inputObj: any) {
  ensureInitSyncGuard();
  const { ptr, len } = encode(inputObj);
  const ret = _exports[funcName](ptr, len) as bigint;
  return decode(ret);
}

await init();

function bind(op: string) { return (input: any) => callJson('invoke_json', { op, input }); }

const manifest = callJson('manifest_json', {});
const api: any = {};
for (const [category, info] of Object.entries(manifest || {})) {
  const methods = info && typeof info === 'object' ? (info as any).methods || {} : {};
  (api as any)[category] = {};
  for (const [name, meta] of Object.entries(methods)) {
    if (!meta || typeof meta !== 'object' || typeof (meta as any).op !== 'string') continue;
    (api as any)[category][name] = bind((meta as any).op);
  }
}

export default api;
