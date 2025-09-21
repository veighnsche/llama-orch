// Placeholder module for applet llm/invoke.
use anyhow::{bail, Result};
use llama_orch_sdk::client::OrchestratorClient;
use serde::{Deserialize, Serialize};

use crate::model::define::ModelRef;
use crate::params::define::Params;

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkMsg {
    pub role: String,
    pub content: String,
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice { pub text: String }

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvokeResult {
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvokeIn {
    pub messages: Vec<SdkMsg>,
    pub model: ModelRef,
    pub params: Params,
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvokeOut { pub result: InvokeResult }

pub fn run(_client: &OrchestratorClient, _input: InvokeIn) -> Result<InvokeOut> {
    // M2 placeholder: SDK networking not implemented yet.
    // Return a deterministic stub so downstream stages can be exercised.
    bail!("unimplemented: OrchestratorClient non-streaming invoke not yet wired")
}
