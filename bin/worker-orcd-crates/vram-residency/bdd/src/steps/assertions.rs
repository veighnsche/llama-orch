//! Assertion step definitions

use cucumber::then;

use super::world::BddWorld;

#[then("the seal should succeed")]
async fn then_seal_should_succeed(world: &mut BddWorld) {
    assert!(
        world.last_succeeded(),
        "Expected seal to succeed, but got error: {:?}",
        world.get_last_error()
    );
    println!("✓ Assertion passed: seal succeeded");
}

#[then("the seal should fail")]
async fn then_seal_should_fail(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected seal to fail, but it succeeded"
    );
    println!("✓ Assertion passed: seal failed as expected");
}

#[then(expr = "the seal should fail with {string}")]
pub async fn then_seal_should_fail_with(world: &mut BddWorld, error_type: String) {
    assert!(
        world.last_failed(),
        "Expected seal to fail with {}, but it succeeded",
        error_type
    );

    let error_msg = world.get_last_error().unwrap_or("");

    match error_type.as_str() {
        "InvalidInput" => {
            assert!(
                error_msg.contains("invalid") || error_msg.contains("Invalid"),
                "Expected InvalidInput error, got: {}",
                error_msg
            );
        }
        "InsufficientVram" => {
            assert!(
                error_msg.contains("insufficient") || error_msg.contains("Insufficient"),
                "Expected InsufficientVram error, got: {}",
                error_msg
            );
        }
        "SealVerificationFailed" => {
            assert!(
                error_msg.contains("verification") || error_msg.contains("Verification"),
                "Expected SealVerificationFailed error, got: {}",
                error_msg
            );
        }
        "NotSealed" => {
            assert!(
                error_msg.contains("not sealed") || error_msg.contains("Not sealed") || error_msg.contains("NotSealed"),
                "Expected NotSealed error, got: {}",
                error_msg
            );
        }
        _ => panic!("Unknown error type: {}", error_type),
    }

    println!("✓ Assertion passed: seal failed with {}", error_type);
}

#[then("the verification should succeed")]
async fn then_verification_should_succeed(world: &mut BddWorld) {
    assert!(
        world.last_succeeded(),
        "Expected verification to succeed, but got error: {:?}",
        world.get_last_error()
    );
    println!("✓ Assertion passed: verification succeeded");
}

#[then("the verification should fail")]
async fn then_verification_should_fail(world: &mut BddWorld) {
    assert!(
        world.last_failed(),
        "Expected verification to fail, but it succeeded"
    );
    println!("✓ Assertion passed: verification failed as expected");
}

#[then(expr = "the verification should fail with {string}")]
async fn then_verification_should_fail_with(world: &mut BddWorld, error_type: String) {
    then_seal_should_fail_with(world, error_type).await;
}

#[then("the sealed shard should have:")]
async fn then_sealed_shard_should_have(world: &mut BddWorld, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected table in step");
    let shard_id = world.shard_id.clone();
    let shard = world.shards.get(&shard_id)
        .expect("No shard found for assertion");

    for row in table.rows.iter().skip(1) {
        // Skip header row
        let field = &row[0];
        let expected = &row[1];

        match field.as_str() {
            "shard_id" => {
                assert_eq!(&shard.shard_id, expected, "shard_id mismatch");
            }
            "gpu_device" => {
                let expected_device: u32 = expected.parse().expect("Invalid gpu_device");
                assert_eq!(shard.gpu_device, expected_device, "gpu_device mismatch");
            }
            "vram_bytes" => {
                let expected_bytes: usize = expected.parse().expect("Invalid vram_bytes");
                assert_eq!(shard.vram_bytes, expected_bytes, "vram_bytes mismatch");
            }
            "digest" => {
                if expected.contains("hex chars") {
                    let expected_len: usize = expected.split_whitespace()
                        .next()
                        .and_then(|s| s.parse().ok())
                        .expect("Invalid digest length");
                    assert_eq!(shard.digest.len(), expected_len, "digest length mismatch");
                } else {
                    assert_eq!(&shard.digest, expected, "digest mismatch");
                }
            }
            _ => panic!("Unknown field: {}", field),
        }
    }

    println!("✓ Assertion passed: sealed shard has expected values");
}

#[then(expr = "the error should indicate needed={int}MB available={int}MB")]
async fn then_error_should_indicate_capacity(
    world: &mut BddWorld,
    needed_mb: usize,
    available_mb: usize,
) {
    let error_msg = world.get_last_error()
        .expect("No error message found");

    let needed_bytes = needed_mb * 1024 * 1024;
    let available_bytes = available_mb * 1024 * 1024;

    assert!(
        error_msg.contains(&needed_bytes.to_string()) || error_msg.contains(&needed_mb.to_string()),
        "Error message should contain needed capacity: {}",
        error_msg
    );

    assert!(
        error_msg.contains(&available_bytes.to_string()) || error_msg.contains(&available_mb.to_string()),
        "Error message should contain available capacity: {}",
        error_msg
    );

    println!("✓ Assertion passed: error indicates capacity correctly");
}

#[then(expr = "an audit event {string} should be emitted")]
async fn then_audit_event_should_be_emitted(_world: &mut BddWorld, event_type: String) {
    // TODO: Implement audit logging verification when audit-logging is integrated
    println!("⚠ Audit logging not yet integrated - skipping {} event check", event_type);
}

#[then(expr = "the event should have severity {string}")]
async fn then_event_should_have_severity(_world: &mut BddWorld, severity: String) {
    // TODO: Implement severity checking when audit-logging is integrated
    println!("⚠ Audit logging not yet integrated - skipping severity {} check", severity);
}

#[then("no audit event should be emitted")]
async fn then_no_audit_event_should_be_emitted(_world: &mut BddWorld) {
    // TODO: Implement audit logging verification when audit-logging is integrated
    println!("⚠ Audit logging not yet integrated - skipping no-event check");
}
