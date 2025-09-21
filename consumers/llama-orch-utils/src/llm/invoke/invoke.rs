// Placeholder module for applet llm/invoke.
use anyhow::{bail, Result};
use llama_orch_sdk::client::OrchestratorClient;

use crate::model::define::ModelRef;
use crate::params::define::Params;

#[derive(Debug, Clone)]
pub struct SdkMsg {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct Choice { pub text: String }

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
}

#[derive(Debug, Clone, Default)]
pub struct InvokeResult {
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone)]
pub struct InvokeIn {
    pub messages: Vec<SdkMsg>,
    pub model: ModelRef,
    pub params: Params,
}

#[derive(Debug, Clone)]
pub struct InvokeOut { pub result: InvokeResult }

pub fn run(_client: &OrchestratorClient, _input: InvokeIn) -> Result<InvokeOut> {
    // M2 placeholder: SDK networking not implemented yet.
    // Return a deterministic stub so downstream stages can be exercised.
    bail!("unimplemented: OrchestratorClient non-streaming invoke not yet wired")
}
