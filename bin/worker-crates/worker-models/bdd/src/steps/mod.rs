//! BDD steps for worker-models

mod world;

pub use world::ModelsWorld;

use cucumber::{given, then, when};

#[given(expr = "a GGUF file {string}")]
async fn given_gguf_file(world: &mut ModelsWorld, filename: String) {
    world.model_file = Some(filename);
}

#[when("I detect the model architecture")]
async fn when_detect_architecture(world: &mut ModelsWorld) {
    // TODO: Use ModelFactory to detect architecture
    world.detected_architecture = Some("llama".to_string()); // Placeholder
}

#[when("I create a model adapter")]
async fn when_create_adapter(world: &mut ModelsWorld) {
    // TODO: Create adapter via factory
    world.adapter_created = true;
}

#[then(expr = "the architecture should be {string}")]
async fn then_architecture_is(world: &mut ModelsWorld, expected: String) {
    let arch = world.detected_architecture.as_ref().expect("no architecture");
    assert_eq!(arch, &expected, "architecture mismatch");
}

#[then(expr = "the adapter should be {string}")]
async fn then_adapter_is(world: &mut ModelsWorld, expected: String) {
    assert!(world.adapter_created, "adapter not created");
    // TODO: Verify adapter type
    let _ = expected; // Placeholder
}

#[then("the adapter should support inference")]
async fn then_supports_inference(world: &mut ModelsWorld) {
    assert!(world.adapter_created, "adapter not created");
    // TODO: Verify adapter has inference methods
}
