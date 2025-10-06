//! BDD steps for worker-models

mod world;

pub use world::ModelsWorld;

use cucumber::{given, then, when};
use worker_gguf::GGUFMetadata;
use worker_models::{AdapterFactory, ModelType};

#[given(expr = "a GGUF file {string}")]
async fn given_gguf_file(world: &mut ModelsWorld, filename: String) {
    world.model_file = Some(filename);
}

#[when("I detect the model architecture")]
async fn when_detect_architecture(world: &mut ModelsWorld) {
    let filename = world.model_file.as_ref().expect("model file not set");
    let metadata = GGUFMetadata::from_file(filename).expect("failed to parse GGUF");
    let arch = metadata.architecture().expect("no architecture");
    world.detected_architecture = Some(arch);
}

#[when("I create a model adapter")]
async fn when_create_adapter(world: &mut ModelsWorld) {
    let filename = world.model_file.as_ref().expect("model file not set");
    let adapter = AdapterFactory::from_gguf(filename).expect("failed to create adapter");
    world.model_type = Some(adapter.model_type());
    world.vocab_size = Some(adapter.vocab_size().expect("vocab_size failed"));
    world.num_layers = Some(adapter.num_layers().expect("num_layers failed"));
    world.adapter_created = true;
}

#[then(expr = "the architecture should be {string}")]
async fn then_architecture_is(world: &mut ModelsWorld, expected: String) {
    let arch = world.detected_architecture.as_ref().expect("no architecture");
    assert_eq!(arch, &expected, "architecture mismatch");
}

#[then(expr = "the adapter should be {string}")]
async fn then_adapter_is(world: &mut ModelsWorld, expected_type: String) {
    assert!(world.adapter_created, "adapter not created");
    let model_type = world.model_type.expect("model type not set");

    // Check if model type matches the architecture family
    match (expected_type.as_str(), model_type) {
        ("LlamaAdapter", ModelType::Qwen2_5) => {}
        ("LlamaAdapter", ModelType::Phi3) => {}
        ("LlamaAdapter", ModelType::Llama2) => {}
        ("LlamaAdapter", ModelType::Llama3) => {}
        ("GPTAdapter", ModelType::GPT2) => {}
        ("GPTAdapter", ModelType::GPT3) => {}
        _ => panic!("Model type mismatch: expected {} but got {:?}", expected_type, model_type),
    }
}

#[then("the adapter should support inference")]
async fn then_supports_inference(world: &mut ModelsWorld) {
    assert!(world.adapter_created, "adapter not created");
    assert!(world.vocab_size.is_some(), "vocab_size not available");
    assert!(world.num_layers.is_some(), "num_layers not available");
}

#[then(expr = "the model should have vocab size {int}")]
async fn then_vocab_size(world: &mut ModelsWorld, expected: usize) {
    let vocab_size = world.vocab_size.expect("vocab_size not set");
    assert_eq!(vocab_size, expected, "vocab size mismatch");
}

#[then(expr = "the model should have {int} layers")]
async fn then_num_layers(world: &mut ModelsWorld, expected: usize) {
    let num_layers = world.num_layers.expect("num_layers not set");
    assert_eq!(num_layers, expected, "num layers mismatch");
}
