use serde::Serialize;
use std::collections::BTreeMap;
use serde_json::Value;

// Bring applet types in scope for parsing/serialization
use crate::fs::file_reader::{ReadRequest, ReadResponse, FileBlob};
use crate::fs::file_writer::{WriteIn, WriteOut};
use crate::prompt::message::{MessageIn, Message};
use crate::prompt::thread::{ThreadIn, ThreadOut};
use crate::model::define::{ModelRef, ModelDefineIn};
use crate::params::define::Params;
use crate::llm::invoke::{InvokeIn, InvokeOut, InvokeResult, SdkMsg, Choice, Usage};

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
