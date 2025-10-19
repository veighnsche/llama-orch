// Placeholder module for applet llm/invoke.
use rbee_sdk::client::OrchestratorClient;
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
pub struct Choice {
    pub text: String,
}

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
pub struct InvokeOut {
    pub result: InvokeResult,
}

pub fn run(
    _client: &OrchestratorClient,
    _input: InvokeIn,
) -> Result<InvokeOut, crate::error::Error> {
    // M2 DRAFT: SDK wiring not implemented; return a typed Unimplemented error (no panic).
    Err(crate::error::Error::Unimplemented {
        message: "unimplemented: llm.invoke requires SDK wiring".to_string(),
    })
}
