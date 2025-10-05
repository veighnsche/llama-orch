//! BDD steps for worker-compute

mod world;

pub use world::ComputeWorld;

use cucumber::{given, then, when};
use worker_compute::{ComputeBackend, ComputeError};

// Mock backend for BDD testing
struct BddBackend;

struct BddContext {
    device_id: i32,
}

struct BddModel {
    path: String,
    memory: u64,
}

struct BddInferenceResult {
    tokens: Vec<String>,
    current: usize,
    max_tokens: usize,
}

impl ComputeBackend for BddBackend {
    type Context = BddContext;
    type Model = BddModel;
    type InferenceResult = BddInferenceResult;

    fn init(device_id: i32) -> Result<Self::Context, ComputeError> {
        if device_id < 0 {
            return Err(ComputeError::DeviceNotFound);
        }
        Ok(BddContext { device_id })
    }

    fn load_model(ctx: &Self::Context, path: &str) -> Result<Self::Model, ComputeError> {
        if path.is_empty() {
            return Err(ComputeError::InvalidParameter("empty path".to_string()));
        }
        if !path.ends_with(".gguf") {
            return Err(ComputeError::ModelLoadFailed("invalid format".to_string()));
        }

        let memory = if path.contains("8b") {
            8_000_000_000
        } else {
            16_000_000_000
        };

        Ok(BddModel {
            path: path.to_string(),
            memory,
        })
    }

    fn inference_start(
        model: &Self::Model,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<Self::InferenceResult, ComputeError> {
        if prompt.is_empty() {
            return Err(ComputeError::InvalidParameter("empty prompt".to_string()));
        }
        if max_tokens == 0 {
            return Err(ComputeError::InvalidParameter("max_tokens must be > 0".to_string()));
        }
        if !(0.0..=2.0).contains(&temperature) {
            return Err(ComputeError::InvalidParameter("temperature out of range".to_string()));
        }

        Ok(BddInferenceResult {
            tokens: vec!["Hello".to_string(), " world".to_string()],
            current: 0,
            max_tokens,
        })
    }

    fn inference_next_token(
        result: &mut Self::InferenceResult,
    ) -> Result<Option<String>, ComputeError> {
        if result.current >= result.max_tokens || result.current >= result.tokens.len() {
            return Ok(None);
        }
        let token = result.tokens[result.current].clone();
        result.current += 1;
        Ok(Some(token))
    }

    fn get_memory_usage(model: &Self::Model) -> u64 {
        model.memory
    }

    fn memory_architecture() -> &'static str {
        "bdd-test"
    }
}

// Device initialization scenarios
#[given(expr = "a compute device with ID {int}")]
async fn given_device_id(world: &mut ComputeWorld, device_id: i32) {
    world.device_id = Some(device_id);
}

#[when("I initialize the compute backend")]
async fn when_initialize_backend(world: &mut ComputeWorld) {
    let device_id = world.device_id.unwrap_or(0);
    match BddBackend::init(device_id) {
        Ok(_) => {
            world.init_success = true;
        }
        Err(e) => {
            world.init_error = Some(e);
        }
    }
}

#[then("the backend should initialize successfully")]
async fn then_init_success(world: &mut ComputeWorld) {
    assert!(world.init_success, "backend initialization failed");
}

#[then("the initialization should fail")]
async fn then_init_fails(world: &mut ComputeWorld) {
    assert!(world.init_error.is_some(), "expected initialization to fail");
}

#[then(expr = "the error should be {string}")]
async fn then_error_is(world: &mut ComputeWorld, error_type: String) {
    let error = world
        .init_error
        .as_ref()
        .or(world.load_error.as_ref())
        .or(world.inference_error.as_ref())
        .expect("no error found");

    match error_type.as_str() {
        "DeviceNotFound" => assert!(matches!(error, ComputeError::DeviceNotFound)),
        "InvalidParameter" => assert!(matches!(error, ComputeError::InvalidParameter(_))),
        "ModelLoadFailed" => assert!(matches!(error, ComputeError::ModelLoadFailed(_))),
        _ => panic!("Unknown error type: {}", error_type),
    }
}

// Model loading scenarios
#[given(expr = "a model path {string}")]
async fn given_model_path(world: &mut ComputeWorld, path: String) {
    world.model_path = Some(path);
}

#[given("an initialized compute backend")]
async fn given_initialized_backend(world: &mut ComputeWorld) {
    world.device_id = Some(0);
    world.init_success = true;
}

#[when("I load the model")]
async fn when_load_model(world: &mut ComputeWorld) {
    let device_id = world.device_id.unwrap_or(0);
    let ctx = BddBackend::init(device_id).unwrap();
    let path = world.model_path.as_ref().expect("model path not set");

    match BddBackend::load_model(&ctx, path) {
        Ok(model) => {
            world.model_loaded = true;
            world.model_memory = Some(BddBackend::get_memory_usage(&model));
        }
        Err(e) => {
            world.load_error = Some(e);
        }
    }
}

#[then("the model should load successfully")]
async fn then_model_loads(world: &mut ComputeWorld) {
    assert!(world.model_loaded, "model loading failed");
}

#[then("the model loading should fail")]
async fn then_model_load_fails(world: &mut ComputeWorld) {
    assert!(world.load_error.is_some(), "expected model loading to fail");
}

#[then(expr = "the memory usage should be {int} bytes")]
async fn then_memory_usage(world: &mut ComputeWorld, expected: u64) {
    let actual = world.model_memory.expect("memory usage not set");
    assert_eq!(actual, expected, "memory usage mismatch");
}

// Inference scenarios
#[given(expr = "a prompt {string}")]
async fn given_prompt(world: &mut ComputeWorld, prompt: String) {
    world.prompt = Some(prompt);
}

#[given(expr = "max_tokens is {int}")]
async fn given_max_tokens(world: &mut ComputeWorld, max_tokens: usize) {
    world.max_tokens = Some(max_tokens);
}

#[given(expr = "temperature is {float}")]
async fn given_temperature(world: &mut ComputeWorld, temperature: f32) {
    world.temperature = Some(temperature);
}

#[given("a loaded model")]
async fn given_loaded_model(world: &mut ComputeWorld) {
    world.model_path = Some("/models/test.gguf".to_string());
    world.model_loaded = true;
}

#[when("I start inference")]
async fn when_start_inference(world: &mut ComputeWorld) {
    let device_id = world.device_id.unwrap_or(0);
    let ctx = BddBackend::init(device_id).unwrap();
    let path = world.model_path.as_ref().expect("model path not set");
    let model = BddBackend::load_model(&ctx, path).unwrap();

    let prompt = world.prompt.as_ref().expect("prompt not set");
    let max_tokens = world.max_tokens.unwrap_or(100);
    let temperature = world.temperature.unwrap_or(0.7);

    match BddBackend::inference_start(&model, prompt, max_tokens, temperature, 42) {
        Ok(mut result) => {
            world.inference_started = true;
            while let Some(token) = BddBackend::inference_next_token(&mut result).unwrap() {
                world.tokens_generated.push(token);
            }
        }
        Err(e) => {
            world.inference_error = Some(e);
        }
    }
}

#[then("inference should start successfully")]
async fn then_inference_starts(world: &mut ComputeWorld) {
    assert!(world.inference_started, "inference failed to start");
}

#[then("the inference should fail")]
async fn then_inference_fails(world: &mut ComputeWorld) {
    assert!(
        world.inference_error.is_some(),
        "expected inference to fail"
    );
}

#[then(expr = "I should receive {int} tokens")]
async fn then_receive_tokens(world: &mut ComputeWorld, expected: usize) {
    assert_eq!(
        world.tokens_generated.len(),
        expected,
        "token count mismatch"
    );
}
