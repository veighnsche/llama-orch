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
use crate::prompt::message::{MessageIn, Message};
use crate::prompt::thread::{ThreadIn, ThreadOut};
use crate::model::define::ModelRef;
use crate::params::define::Params;
use crate::llm::invoke::{InvokeIn, InvokeResult};

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    // Allocate a buffer with the requested capacity and leak it to the caller.
    let mut buf: Vec<u8> = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    mem::forget(buf);
    ptr
}

// ---------- prompt.message ----------
#[no_mangle]
pub extern "C" fn prompt_message_json(req_ptr: *const u8, req_len: usize) -> u64 {
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    let req: MessageIn = match serde_json::from_slice(req_bytes) {
        Ok(v) => v,
        Err(e) => return leak_exact(format!("{{\"error\":\"invalid_request\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0,
    };
    let resp: Message = match crate::prompt::message::run(req) { Ok(v) => v, Err(e) => return leak_exact(format!("{{\"error\":\"io_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 };
    match serde_json::to_vec(&resp) { Ok(v) => leak_exact(&v).0, Err(e) => leak_exact(format!("{{\"error\":\"serialize_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 }
}

// ---------- prompt.thread ----------
#[no_mangle]
pub extern "C" fn prompt_thread_json(req_ptr: *const u8, req_len: usize) -> u64 {
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    let req: ThreadIn = match serde_json::from_slice(req_bytes) {
        Ok(v) => v,
        Err(e) => return leak_exact(format!("{{\"error\":\"invalid_request\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0,
    };
    let resp: ThreadOut = match crate::prompt::thread::run(req) { Ok(v) => v, Err(e) => return leak_exact(format!("{{\"error\":\"io_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 };
    match serde_json::to_vec(&resp) { Ok(v) => leak_exact(&v).0, Err(e) => leak_exact(format!("{{\"error\":\"serialize_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 }
}

// ---------- model.define ----------
#[derive(Deserialize)]
struct ModelDefineIn { model_id: String, engine_id: Option<String>, pool_hint: Option<String> }

#[no_mangle]
pub extern "C" fn model_define_json(req_ptr: *const u8, req_len: usize) -> u64 {
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    let req: ModelDefineIn = match serde_json::from_slice(req_bytes) { Ok(v) => v, Err(e) => return leak_exact(format!("{{\"error\":\"invalid_request\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 };
    let resp: ModelRef = crate::model::define::run(req.model_id, req.engine_id, req.pool_hint);
    match serde_json::to_vec(&resp) { Ok(v) => leak_exact(&v).0, Err(e) => leak_exact(format!("{{\"error\":\"serialize_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 }
}

// ---------- params.define ----------
#[no_mangle]
pub extern "C" fn params_define_json(req_ptr: *const u8, req_len: usize) -> u64 {
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    let req: Params = match serde_json::from_slice(req_bytes) { Ok(v) => v, Err(e) => return leak_exact(format!("{{\"error\":\"invalid_request\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 };
    let resp: Params = crate::params::define::run(req);
    match serde_json::to_vec(&resp) { Ok(v) => leak_exact(&v).0, Err(e) => leak_exact(format!("{{\"error\":\"serialize_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 }
}

// ---------- llm.invoke ----------
#[no_mangle]
pub extern "C" fn llm_invoke_json(req_ptr: *const u8, req_len: usize) -> u64 {
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    let req: InvokeIn = match serde_json::from_slice(req_bytes) { Ok(v) => v, Err(e) => return leak_exact(format!("{{\"error\":\"invalid_request\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 };
    let client = llama_orch_sdk::client::OrchestratorClient::default();
    match crate::llm::invoke::run(&client, req) {
        Ok(v) => match serde_json::to_vec(&v) { Ok(v) => leak_exact(&v).0, Err(e) => leak_exact(format!("{{\"error\":\"serialize_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 },
        Err(e) => leak_exact(format!("{{\"error\":\"unimplemented\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0,
    }
}

// ---------- orch.response_extractor ----------
#[no_mangle]
pub extern "C" fn orch_response_extractor_json(req_ptr: *const u8, req_len: usize) -> u64 {
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    let req: InvokeResult = match serde_json::from_slice(req_bytes) { Ok(v) => v, Err(e) => return leak_exact(format!("{{\"error\":\"invalid_request\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 };
    let s = crate::orch::response_extractor::run(&req);
    // Return as JSON string value
    match serde_json::to_vec(&s) { Ok(v) => leak_exact(&v).0, Err(e) => leak_exact(format!("{{\"error\":\"serialize_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0 }
}

#[no_mangle]
pub extern "C" fn fs_write_file_json(req_ptr: *const u8, req_len: usize) -> u64 {
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    let req: WriteIn = match serde_json::from_slice(req_bytes) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("{{\"error\":\"invalid_request\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap());
            return leak_exact(msg.as_bytes()).0;
        }
    };

    let resp: WriteOut = match crate::fs::file_writer::run(req) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("{{\"error\":\"fs_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap());
            return leak_exact(msg.as_bytes()).0;
        }
    };

    match serde_json::to_vec(&resp) {
        Ok(v) => leak_exact(&v).0,
        Err(e) => leak_exact(format!("{{\"error\":\"serialize_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0,
    }
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

#[no_mangle]
pub extern "C" fn fs_read_file_json(req_ptr: *const u8, req_len: usize) -> u64 {
    // SAFETY: The memory is provided by the host; just read a slice for JSON.
    let req_bytes = unsafe { core::slice::from_raw_parts(req_ptr, req_len) };
    let req: ReadRequest = match serde_json::from_slice(req_bytes) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("{{\"error\":\"invalid_request\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap());
            return leak_exact(msg.as_bytes()).0;
        }
    };

    let resp: ReadResponse = match crate::fs::file_reader::run(req) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("{{\"error\":\"fs_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap());
            return leak_exact(msg.as_bytes()).0;
        }
    };

    match serde_json::to_vec(&resp) {
        Ok(v) => leak_exact(&v).0,
        Err(e) => leak_exact(format!("{{\"error\":\"serialize_error\",\"message\":{}}}", serde_json::to_string(&e.to_string()).unwrap()).as_bytes()).0,
    }
}
