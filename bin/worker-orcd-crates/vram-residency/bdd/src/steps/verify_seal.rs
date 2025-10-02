//! Step definitions for seal verification operations

use cucumber::{given, when};

use super::world::BddWorld;

#[given(expr = "a sealed shard {string} with {int}MB of data")]
pub async fn given_sealed_shard(world: &mut BddWorld, shard_id: String, size_mb: usize) {
    // First create the manager and model data
    let capacity_mb = size_mb * 2; // Ensure enough capacity
    super::seal_model::given_vram_manager_with_capacity(world, capacity_mb).await;
    super::seal_model::given_model_with_data(world, size_mb).await;

    // Store shard_id before sealing
    world.shard_id = shard_id.clone();
    
    // Seal the model
    super::seal_model::when_seal_model(world, shard_id.clone(), 0).await;

    if !world.last_succeeded() {
        panic!("Failed to create sealed shard: {:?}", world.get_last_error());
    }

    println!("✓ Sealed shard '{}' created for testing", shard_id);
}

#[given(expr = "the shard digest is modified to {string}")]
async fn given_shard_digest_modified(world: &mut BddWorld, new_digest: String) {
    let shard_id = world.shard_id.clone();
    if let Some(shard) = world.shards.get_mut(&shard_id) {
        shard.digest = new_digest.clone();
        println!("✓ Shard digest modified to: {}", new_digest);
    } else {
        panic!("No shard found with ID: {}", shard_id);
    }
}

#[given("the shard signature is replaced with zeros")]
async fn given_shard_signature_zeroed(world: &mut BddWorld) {
    let shard_id = world.shard_id.clone();
    if let Some(shard) = world.shards.get_mut(&shard_id) {
        // Replace signature with zeros (invalid signature)
        shard.replace_signature_for_test(vec![0u8; 32]);
        println!("✓ Shard '{}' signature replaced with zeros", shard_id);
    } else {
        panic!("No shard found with ID: {}", shard_id);
    }
}

#[when("I verify the sealed shard")]
pub async fn when_verify_sealed_shard(world: &mut BddWorld) {
    let shard_id = world.shard_id.clone();
    let shard = world.shards.get(&shard_id)
        .expect("No shard found for verification");

    let manager = world.manager.as_ref()
        .expect("VramManager not initialized");

    let result = manager.verify_sealed(shard);

    match result {
        Ok(()) => {
            println!("✓ Seal verification passed");
            world.store_result(Ok(()));
        }
        Err(e) => {
            println!("✗ Seal verification failed: {}", e);
            world.store_result(Err(e));
        }
    }
}
