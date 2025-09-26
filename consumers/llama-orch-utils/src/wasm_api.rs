// WASM/WASI export surface using a simple JSON-in/JSON-out ABI.
// - Memory management: JS calls `alloc(len)` to get a pointer, writes the request bytes,
//   then calls an exported function which returns a 64-bit BigInt encoding (ptr | (len<<32)).
//   JS reads the response bytes from memory and then calls `free(ptr, cap)` to free them.
// - Target: wasm32-wasip1-threads.

#![allow(clippy::missing_safety_doc)]

use core::mem;
use serde::Deserialize;

// Reuse core types
use crate::fs::file_reader::{ReadRequest, ReadResponse};
use crate::fs::file_writer::{WriteIn, WriteOut};
use crate::llm::invoke::{InvokeIn, InvokeResult};
use crate::model::define::{ModelDefineIn, ModelRef};
use crate::params::define::Params;
use crate::prompt::message::{Message, MessageIn};
use crate::prompt::thread::{ThreadIn, ThreadOut};

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    // Allocate a buffer with the requested capacity and leak it to the caller.
    let mut buf: Vec<u8> = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    mem::forget(buf);
    ptr
}

#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, capacity: usize) {
    // SAFETY: The pointer must have been allocated by `alloc` with the same capacity.
    if ptr.is_null() || capacity == 0 {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, capacity);
    }
}

fn pack_ptr_len(ptr: *mut u8, len: usize) -> u64 {
    let p = ptr as u32 as u64;
    let l = (len as u32 as u64) << 32;
    l | p
}

// Allocate a new Vec<u8> with capacity exactly equal to length and leak it, returning (ptr,len)
fn leak_exact(bytes: &[u8]) -> (u64, *mut u8, usize) {
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    out.extend_from_slice(bytes);
    let len = out.len();
    let ptr = out.as_mut_ptr();
    mem::forget(out);
    (pack_ptr_len(ptr, len), ptr, len)
}

// ---- unified manifest + dispatcher ----
use crate::manifest::make_manifest;

#[no_mangle]
pub extern "C" fn manifest_json(_req_ptr: *const u8, _req_len: usize) -> u64 {
    match serde_json::to_vec(&make_manifest()) {
        Ok(v) => leak_exact(&v).0,
        Err(e) => {
            leak_exact(
                format!(
                    "{{\"error\":\"serialize_error\",\"message\":{}}}",
                    serde_json::to_string(&e.to_string()).unwrap()
                )
                .as_bytes(),
            )
            .0
        }
    }
}

#[derive(Deserialize)]
struct InvokeReq<T> {
    op: String,
    input: T,
}

// ModelDefineIn provided by crate::model::define

#[no_mangle]
pub extern "C" fn invoke_json(req_ptr: *const u8, req_len: usize) -> u64 {
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    // First, parse to get the op string; we'll re-parse input per op with the right type.
    let v: serde_json::Value = match serde_json::from_slice(req_bytes) {
        Ok(v) => v,
        Err(e) => {
            return leak_exact(
                format!(
                    "{{\"error\":\"invalid_request\",\"message\":{}}}",
                    serde_json::to_string(&e.to_string()).unwrap()
                )
                .as_bytes(),
            )
            .0
        }
    };
    let op = match v.get("op").and_then(|x| x.as_str()) {
        Some(s) => s,
        None => return leak_exact(b"{\"error\":\"invalid_request\",\"message\":\"missing op\"}").0,
    };
    let input = match v.get("input") {
        Some(x) => x,
        None => {
            return leak_exact(b"{\"error\":\"invalid_request\",\"message\":\"missing input\"}").0
        }
    };
    let bytes = crate::manifest::dispatch(op, input.clone());
    leak_exact(&bytes).0
}
