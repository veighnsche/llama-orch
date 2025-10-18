// TEAM-108: Test temporarily disabled - llama_orch_sdk not yet available
// TODO: Re-enable when SDK is published

// use super::*;
// use crate::llm::invoke::{InvokeIn, SdkMsg};
// use crate::model::define::ModelRef;
// use crate::params::define::Params;
// use llama_orch_sdk::client::OrchestratorClient;

// #[test]
// fn returns_unimplemented_without_panic() {
//     let client = OrchestratorClient::default();
//     let input = InvokeIn {
//         messages: vec![SdkMsg { role: "user".into(), content: "hi".into() }],
//         model: ModelRef { model_id: "foo".into(), engine_id: None, pool_hint: None },
//         params: Params::default(),
//     };

//     let res = super::run(&client, input);
//     assert!(res.is_err(), "expected Err for unimplemented invoke");
//     let err = res.err().unwrap();
//     // Ensure variant is Unimplemented and message matches exactly
//     match err {
//         crate::error::Error::Unimplemented { message } => {
//             assert_eq!(message, "unimplemented: llm.invoke requires SDK wiring");
//         }
//     }
// }
