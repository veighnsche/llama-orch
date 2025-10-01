//! Step definitions for seal_model operations

use cucumber::{given, when};
use gpu_info;
use vram_residency::VramManager;

use super::world::BddWorld;

#[given(expr = "a VramManager with {int}MB capacity")]
pub async fn given_vram_manager_with_capacity(world: &mut BddWorld, capacity_mb: usize) {
    // Detect GPU
    let gpu_info = gpu_info::detect_gpus();
    world.gpu_info = Some(gpu_info);

    // Create VramManager (auto-detects GPU)
    world.manager = Some(VramManager::new());
    world.vram_capacity = capacity_mb * 1024 * 1024;

    println!("✓ VramManager created with {}MB capacity (mode: {})", 
        capacity_mb, world.test_mode());
}

#[given(expr = "a model with {int}MB of data")]
pub async fn given_model_with_data(world: &mut BddWorld, size_mb: usize) {
    let size_bytes = size_mb * 1024 * 1024;
    world.model_data = vec![0u8; size_bytes];
    println!("✓ Model data created: {} bytes", size_bytes);
}

#[given(expr = "a model with {int} bytes of data")]
async fn given_model_with_bytes(world: &mut BddWorld, size_bytes: usize) {
    world.model_data = vec![0u8; size_bytes];
    println!("✓ Model data created: {} bytes", size_bytes);
}

#[when(expr = "I seal the model with shard_id {string} on GPU {int}")]
pub async fn when_seal_model(world: &mut BddWorld, shard_id: String, gpu_device: u32) {
    world.shard_id = shard_id.clone();
    world.gpu_device = gpu_device;

    let manager = world.manager.as_mut()
        .expect("VramManager not initialized");

    let result = manager.seal_model(&world.model_data, gpu_device);

    match result {
        Ok(shard) => {
            println!("✓ Model sealed: shard_id={}, vram_bytes={}, digest={}", 
                shard.shard_id, shard.vram_bytes, &shard.digest[..16]);
            world.shards.insert(shard_id, shard);
            world.store_result(Ok(()));
        }
        Err(e) => {
            println!("✗ Seal failed: {}", e);
            world.store_result(Err(e));
        }
    }
}

#[when("I seal the model")]
async fn when_seal_model_default(world: &mut BddWorld) {
    when_seal_model(world, world.shard_id.clone(), world.gpu_device).await;
}
