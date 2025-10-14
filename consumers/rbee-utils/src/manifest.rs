use serde::Serialize;
#[allow(unused_imports)]
use serde_json::Value;
use std::collections::BTreeMap;

// Bring applet types in scope for parsing/serialization (also used by TS export helpers)
#[allow(unused_imports)]
use crate::fs::file_reader::{FileBlob, ReadRequest, ReadResponse};
#[allow(unused_imports)]
use crate::fs::file_writer::{WriteIn, WriteOut};
#[allow(unused_imports)]
use crate::llm::invoke::{Choice, InvokeIn, InvokeOut, InvokeResult, SdkMsg, Usage};
#[allow(unused_imports)]
use crate::model::define::{ModelDefineIn, ModelRef};
#[allow(unused_imports)]
use crate::params::define::Params;
#[allow(unused_imports)]
use crate::prompt::message::{Message, MessageIn};
#[allow(unused_imports)]
use crate::prompt::thread::{ThreadIn, ThreadOut};

#[derive(Serialize)]
pub struct MethodInfo {
    pub op: &'static str,
    pub input: &'static str,
    pub output: &'static str,
}

#[derive(Serialize)]
pub struct Category {
    pub methods: BTreeMap<&'static str, MethodInfo>,
}

#[derive(Serialize)]
pub struct Manifest {
    pub fs: Category,
    pub prompt: Category,
    pub model: Category,
    pub params: Category,
    pub llm: Category,
    pub orch: Category,
}

pub fn make_manifest() -> Manifest {
    let mut fs = BTreeMap::new();
    fs.insert(
        "readFile",
        MethodInfo { op: "fs_read_file_json", input: "ReadRequest", output: "ReadResponse" },
    );
    fs.insert(
        "writeFile",
        MethodInfo { op: "fs_write_file_json", input: "WriteIn", output: "WriteOut" },
    );

    let mut prompt = BTreeMap::new();
    prompt.insert(
        "message",
        MethodInfo { op: "prompt_message_json", input: "MessageIn", output: "Message" },
    );
    prompt.insert(
        "thread",
        MethodInfo { op: "prompt_thread_json", input: "ThreadIn", output: "ThreadOut" },
    );

    let mut model = BTreeMap::new();
    model.insert(
        "define",
        MethodInfo { op: "model_define_json", input: "ModelDefineIn", output: "ModelRef" },
    );

    let mut params = BTreeMap::new();
    params.insert(
        "define",
        MethodInfo { op: "params_define_json", input: "Params", output: "Params" },
    );

    let mut llm = BTreeMap::new();
    llm.insert(
        "invoke",
        MethodInfo { op: "llm_invoke_json", input: "InvokeIn", output: "InvokeOut" },
    );

    let mut orch = BTreeMap::new();
    orch.insert(
        "responseExtractor",
        MethodInfo { op: "orch_response_extractor_json", input: "InvokeResult", output: "string" },
    );

    Manifest {
        fs: Category { methods: fs },
        prompt: Category { methods: prompt },
        model: Category { methods: model },
        params: Category { methods: params },
        llm: Category { methods: llm },
        orch: Category { methods: orch },
    }
}

pub fn manifest_json_string() -> Result<String, serde_json::Error> {
    serde_json::to_string(&make_manifest())
}

/// Append TypeScript type declarations for the applet manifest and request/response shapes.
/// This is a minimal stub to satisfy CI; full TS generation will be added in a follow-up.
pub fn append_ts_types(buf: &mut String) {
    // Manifest types
    buf.push_str("export type MethodInfo = { op: string; input: string; output: string };\n");
    buf.push_str("export type Category = { methods: Record<string, MethodInfo> };\n");
    buf.push_str(
        "export type Manifest = { fs: Category; prompt: Category; model: Category; params: Category; llm: Category; orch: Category };\n\n",
    );
    // Placeholders for request/response types referenced in the manifest; concrete TS will replace these later.
    buf.push_str(
        "// Placeholder request/response declarations (to be replaced with concrete TS)\n",
    );
    buf.push_str("export type ReadRequest = unknown;\nexport type ReadResponse = unknown;\n");
    buf.push_str("export type WriteIn = unknown;\nexport type WriteOut = unknown;\n");
    buf.push_str("export type MessageIn = unknown;\nexport type Message = unknown;\n");
    buf.push_str("export type ThreadIn = unknown;\nexport type ThreadOut = unknown;\n");
    buf.push_str("export type ModelDefineIn = unknown;\nexport type ModelRef = unknown;\n");
    buf.push_str("export type Params = unknown;\n");
    buf.push_str("export type InvokeIn = unknown;\nexport type InvokeOut = unknown;\nexport type InvokeResult = unknown;\n");
    buf.push_str("export type SdkMsg = unknown;\nexport type Usage = unknown;\n");
}
