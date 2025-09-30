#[cfg(feature = "provider-llamacpp")]
#[path = "llamacpp.rs"]
pub mod llamacpp;

#[cfg(feature = "provider-tgi")]
pub mod tgi;

#[cfg(feature = "provider-triton")]
pub mod triton;

#[cfg(feature = "provider-vllm")]
pub mod vllm;
