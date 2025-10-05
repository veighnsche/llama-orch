//! BDD steps for worker-gguf

mod world;

pub use world::GGUFWorld;

use cucumber::{given, then, when};
use worker_gguf::GGUFMetadata;

#[given(expr = "a GGUF file {string}")]
async fn given_gguf_file(world: &mut GGUFWorld, filename: String) {
    world.filename = Some(filename);
}

#[when("I parse the GGUF metadata")]
async fn when_parse_metadata(world: &mut GGUFWorld) {
    let filename = world.filename.as_ref().expect("filename not set");
    world.metadata = Some(GGUFMetadata::from_file(filename).expect("failed to parse GGUF"));
}

#[then(expr = "the architecture should be {string}")]
async fn then_architecture_should_be(world: &mut GGUFWorld, expected: String) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    let arch = metadata.architecture().expect("no architecture");
    assert_eq!(arch, expected, "architecture mismatch");
}

#[then(expr = "the vocabulary size should be {int}")]
async fn then_vocab_size_should_be(world: &mut GGUFWorld, expected: usize) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    let vocab_size = metadata.vocab_size().expect("no vocab size");
    assert_eq!(vocab_size, expected, "vocab size mismatch");
}

#[then(expr = "the hidden dimension should be {int}")]
async fn then_hidden_dim_should_be(world: &mut GGUFWorld, expected: usize) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    let hidden_dim = metadata.hidden_dim().expect("no hidden dim");
    assert_eq!(hidden_dim, expected, "hidden dim mismatch");
}

#[then(expr = "the number of layers should be {int}")]
async fn then_num_layers_should_be(world: &mut GGUFWorld, expected: usize) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    let num_layers = metadata.num_layers().expect("no num layers");
    assert_eq!(num_layers, expected, "num layers mismatch");
}

#[then(expr = "the number of attention heads should be {int}")]
async fn then_num_heads_should_be(world: &mut GGUFWorld, expected: usize) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    let num_heads = metadata.num_heads().expect("no num heads");
    assert_eq!(num_heads, expected, "num heads mismatch");
}

#[then(expr = "the number of KV heads should be {int}")]
async fn then_num_kv_heads_should_be(world: &mut GGUFWorld, expected: usize) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    let num_kv_heads = metadata.num_kv_heads().expect("no num kv heads");
    assert_eq!(num_kv_heads, expected, "num kv heads mismatch");
}

#[then("the model should use GQA")]
async fn then_should_use_gqa(world: &mut GGUFWorld) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    assert!(metadata.is_gqa(), "model should use GQA");
}

#[then("the model should use MHA")]
async fn then_should_use_mha(world: &mut GGUFWorld) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    assert!(!metadata.is_gqa(), "model should use MHA");
}

#[then(expr = "the RoPE frequency base should be {float}")]
async fn then_rope_freq_base_should_be(world: &mut GGUFWorld, expected: f32) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    let rope_freq_base = metadata.rope_freq_base().expect("no rope freq base");
    assert_eq!(rope_freq_base, expected, "rope freq base mismatch");
}

#[then(expr = "the context length should be {int}")]
async fn then_context_length_should_be(world: &mut GGUFWorld, expected: usize) {
    let metadata = world.metadata.as_ref().expect("metadata not parsed");
    let context_length = metadata.context_length().expect("no context length");
    assert_eq!(context_length, expected, "context length mismatch");
}
