//! BDD steps for worker-gguf

mod world;

pub use world::GGUFWorld;

use cucumber::{given, then, when};
// TODO: Uncomment after worker-gguf extraction
// use worker_gguf::GGUFMetadata;

#[given(expr = "a GGUF file {string}")]
async fn given_gguf_file(world: &mut GGUFWorld, filename: String) {
    world.filename = Some(filename);
}

#[when("I parse the GGUF metadata")]
async fn when_parse_metadata(world: &mut GGUFWorld) {
    let _filename = world.filename.as_ref().expect("filename not set");
    // TODO: Uncomment after worker-gguf extraction
    // world.metadata = Some(GGUFMetadata::from_file(filename).expect("failed to parse GGUF"));
    world.metadata = Some(()); // Placeholder
}

#[then(expr = "the architecture should be {string}")]
async fn then_architecture_should_be(world: &mut GGUFWorld, expected: String) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
    // let arch = metadata.architecture().expect("no architecture");
    // assert_eq!(arch, expected, "architecture mismatch");
    let _ = expected; // Placeholder
}

#[then(expr = "the vocabulary size should be {int}")]
async fn then_vocab_size_should_be(world: &mut GGUFWorld, expected: usize) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
    let _ = expected; // Placeholder
}

#[then(expr = "the hidden dimension should be {int}")]
async fn then_hidden_dim_should_be(world: &mut GGUFWorld, expected: usize) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
    let _ = expected; // Placeholder
}

#[then(expr = "the number of layers should be {int}")]
async fn then_num_layers_should_be(world: &mut GGUFWorld, expected: usize) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
    let _ = expected; // Placeholder
}

#[then(expr = "the number of attention heads should be {int}")]
async fn then_num_heads_should_be(world: &mut GGUFWorld, expected: usize) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
    let _ = expected; // Placeholder
}

#[then(expr = "the number of KV heads should be {int}")]
async fn then_num_kv_heads_should_be(world: &mut GGUFWorld, expected: usize) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
    let _ = expected; // Placeholder
}

#[then("the model should use GQA")]
async fn then_should_use_gqa(world: &mut GGUFWorld) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
}

#[then("the model should use MHA")]
async fn then_should_use_mha(world: &mut GGUFWorld) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
}

#[then(expr = "the RoPE frequency base should be {float}")]
async fn then_rope_freq_base_should_be(world: &mut GGUFWorld, expected: f32) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
    let _ = expected; // Placeholder
}

#[then(expr = "the context length should be {int}")]
async fn then_context_length_should_be(world: &mut GGUFWorld, expected: usize) {
    let _metadata = world.metadata.as_ref().expect("metadata not parsed");
    // TODO: Uncomment after worker-gguf extraction
    let _ = expected; // Placeholder
}
