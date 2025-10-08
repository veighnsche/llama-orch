//! CPU Inference Backend
//!
//! Implements the InferenceBackend trait from worker-http
//!
//! IMPORTS: worker-http, worker-common, worker-tokenizer
//! CHECKPOINT: 0 (Foundation)

mod cpu_backend;

pub use cpu_backend::CpuInferenceBackend;
