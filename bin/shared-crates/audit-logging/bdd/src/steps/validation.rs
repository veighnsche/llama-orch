//! Validation step definitions
//!
//! This module provides Cucumber step definitions for BDD testing of audit event
//! validation with focus on security (log injection prevention).

use audit_logging::{ActorInfo, AuditEvent, AuthMethod, ResourceInfo};
use chrono::Utc;
use cucumber::{given, when};
use std::net::IpAddr;

use super::world::BddWorld;

// ========== Given steps - setup ==========

#[given(expr = "a user ID {string}")]
async fn given_user_id(world: &mut BddWorld, user_id: String) {
    world.user_id = interpret_escape_sequences(&user_id);
}

#[given(expr = "an IP address {string}")]
async fn given_ip_address(world: &mut BddWorld, ip: String) {
    world.ip_addr = ip.parse::<IpAddr>().ok();
}

#[given(expr = "a session ID {string}")]
async fn given_session_id(world: &mut BddWorld, session_id: String) {
    world.session_id = Some(interpret_escape_sequences(&session_id));
}

#[given(expr = "a resource ID {string}")]
async fn given_resource_id(world: &mut BddWorld, resource_id: String) {
    world.resource_id = interpret_escape_sequences(&resource_id);
}

#[given(expr = "a resource type {string}")]
async fn given_resource_type(world: &mut BddWorld, resource_type: String) {
    world.resource_type = interpret_escape_sequences(&resource_type);
}

#[given(expr = "a reason {string}")]
async fn given_reason(world: &mut BddWorld, reason: String) {
    world.reason = interpret_escape_sequences(&reason);
}

#[given(expr = "a details string {string}")]
async fn given_details(world: &mut BddWorld, details: String) {
    world.details = interpret_escape_sequences(&details);
}

#[given(expr = "a path {string}")]
async fn given_path(world: &mut BddWorld, path: String) {
    world.path = interpret_escape_sequences(&path);
}

#[given(expr = "an endpoint {string}")]
async fn given_endpoint(world: &mut BddWorld, endpoint: String) {
    world.endpoint = interpret_escape_sequences(&endpoint);
}

#[given(expr = "a task ID {string}")]
async fn given_task_id(world: &mut BddWorld, task_id: String) {
    world.task_id = interpret_escape_sequences(&task_id);
}

#[given(expr = "a pool ID {string}")]
async fn given_pool_id(world: &mut BddWorld, pool_id: String) {
    world.pool_id = interpret_escape_sequences(&pool_id);
}

#[given(expr = "a node ID {string}")]
async fn given_node_id(world: &mut BddWorld, node_id: String) {
    world.node_id = interpret_escape_sequences(&node_id);
}

#[given(expr = "a model reference {string}")]
async fn given_model_ref(world: &mut BddWorld, model_ref: String) {
    world.model_ref = interpret_escape_sequences(&model_ref);
}

#[given(expr = "a worker ID {string}")]
async fn given_worker_id(world: &mut BddWorld, worker_id: String) {
    world.worker_id = interpret_escape_sequences(&worker_id);
}

#[given(expr = "a shard ID {string}")]
async fn given_shard_id(world: &mut BddWorld, shard_id: String) {
    world.shard_id = interpret_escape_sequences(&shard_id);
}

#[given(expr = "a customer ID {string}")]
async fn given_customer_id(world: &mut BddWorld, customer_id: String) {
    world.customer_id = interpret_escape_sequences(&customer_id);
}

#[given(expr = "a resource ID {string}")]
async fn given_resource_id_step(world: &mut BddWorld, resource_id: String) {
    world.resource_id = interpret_escape_sequences(&resource_id);
}

#[given(expr = "a resource type {string}")]
async fn given_resource_type_step(world: &mut BddWorld, resource_type: String) {
    world.resource_type = interpret_escape_sequences(&resource_type);
}

#[given(expr = "an action {string}")]
async fn given_action(world: &mut BddWorld, action: String) {
    world.endpoint = interpret_escape_sequences(&action); // Reuse endpoint field
}

#[given(expr = "a subject {string}")]
async fn given_subject(world: &mut BddWorld, subject: String) {
    world.user_id = interpret_escape_sequences(&subject); // Reuse for subject
}

#[given(expr = "a job ID {string}")]
async fn given_job_id(world: &mut BddWorld, job_id: String) {
    world.task_id = interpret_escape_sequences(&job_id); // Reuse task_id
}

#[given(expr = "an access type {string}")]
async fn given_access_type(world: &mut BddWorld, access_type: String) {
    world.details = interpret_escape_sequences(&access_type); // Reuse details
}

#[given(expr = "a requester {string}")]
async fn given_requester(world: &mut BddWorld, requester: String) {
    world.user_id = interpret_escape_sequences(&requester); // Reuse user_id
}

#[given(expr = "an export format {string}")]
async fn given_export_format(world: &mut BddWorld, format: String) {
    world.details = interpret_escape_sequences(&format); // Reuse details
}

#[given(expr = "a token fingerprint {string}")]
async fn given_token_fingerprint(world: &mut BddWorld, fingerprint: String) {
    world.details = interpret_escape_sequences(&fingerprint); // Reuse details
}

#[given(expr = "a user ID with {int} characters")]
async fn given_long_user_id(world: &mut BddWorld, length: usize) {
    world.user_id = "A".repeat(length);
}

/// Helper function to interpret escape sequences in Gherkin strings
fn interpret_escape_sequences(s: &str) -> String {
    s.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace("\\0", "\0")
        .replace("\\x1b", "\x1b")
        .replace("\\x01", "\x01")
        .replace("\\x07", "\x07")
        .replace("\\x08", "\x08")
        .replace("\\x0b", "\x0b")
        .replace("\\x0c", "\x0c")
        .replace("\\x1f", "\x1f")
        .replace("\\\\", "\\")
}

// ========== When steps - actions ==========

#[when("I create an AuthSuccess event")]
async fn when_create_auth_success(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::AuthSuccess {
        timestamp: world.now(),
        actor,
        method: AuthMethod::BearerToken,
        path: world.path.clone(),
        service_id: "test-service".to_string(),
    };
    world.current_event = Some(event);
}

#[when("I create an AuthFailure event")]
async fn when_create_auth_failure(world: &mut BddWorld) {
    let event = AuditEvent::AuthFailure {
        timestamp: world.now(),
        attempted_user: Some(world.user_id.clone()),
        reason: world.reason.clone(),
        ip: world.ip_addr.unwrap_or("127.0.0.1".parse().unwrap()),
        path: world.path.clone(),
        service_id: "test-service".to_string(),
    };
    world.current_event = Some(event);
}

#[when("I create a PoolCreated event")]
async fn when_create_pool_created(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::PoolCreated {
        timestamp: world.now(),
        actor,
        pool_id: world.pool_id.clone(),
        model_ref: world.model_ref.clone(),
        node_id: world.node_id.clone(),
        replicas: 1,
        gpu_devices: vec![0],
    };
    world.current_event = Some(event);
}

#[when("I create a PoolDeleted event")]
async fn when_create_pool_deleted(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::PoolDeleted {
        timestamp: world.now(),
        actor,
        pool_id: world.pool_id.clone(),
        model_ref: world.model_ref.clone(),
        node_id: world.node_id.clone(),
        reason: world.reason.clone(),
        replicas_terminated: 1,
    };
    world.current_event = Some(event);
}

#[when("I create a TaskSubmitted event")]
async fn when_create_task_submitted(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::TaskSubmitted {
        timestamp: world.now(),
        actor,
        task_id: world.task_id.clone(),
        model_ref: world.model_ref.clone(),
        prompt_length: 100,
        prompt_hash: "abc123".to_string(),
        max_tokens: 100,
    };
    world.current_event = Some(event);
}

#[when("I create a VramSealed event")]
async fn when_create_vram_sealed(world: &mut BddWorld) {
    let event = AuditEvent::VramSealed {
        timestamp: world.now(),
        shard_id: world.shard_id.clone(),
        gpu_device: 0,
        vram_bytes: 1024,
        digest: "abc123".to_string(),
        worker_id: world.worker_id.clone(),
    };
    world.current_event = Some(event);
}

#[when("I create a PathTraversalAttempt event")]
async fn when_create_path_traversal(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::PathTraversalAttempt {
        timestamp: world.now(),
        actor,
        attempted_path: world.path.clone(),
        endpoint: world.endpoint.clone(),
    };
    world.current_event = Some(event);
}

#[when("I create an AuthorizationGranted event")]
async fn when_create_authorization_granted(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::AuthorizationGranted {
        timestamp: world.now(),
        actor,
        resource: audit_logging::ResourceInfo {
            resource_type: world.resource_type.clone(),
            resource_id: world.resource_id.clone(),
            parent_id: None,
        },
        action: world.endpoint.clone(),
    };
    world.current_event = Some(event);
}

#[when("I create an AuthorizationDenied event")]
async fn when_create_authorization_denied(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::AuthorizationDenied {
        timestamp: world.now(),
        actor,
        resource: audit_logging::ResourceInfo {
            resource_type: world.resource_type.clone(),
            resource_id: world.resource_id.clone(),
            parent_id: None,
        },
        action: world.endpoint.clone(),
        reason: world.reason.clone(),
    };
    world.current_event = Some(event);
}

#[when("I create a PermissionChanged event")]
async fn when_create_permission_changed(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::PermissionChanged {
        timestamp: world.now(),
        actor,
        subject: world.user_id.clone(),
        old_permissions: vec!["read".to_string()],
        new_permissions: vec!["read".to_string(), "write".to_string()],
    };
    world.current_event = Some(event);
}

#[when("I create a TokenCreated event")]
async fn when_create_token_created(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::TokenCreated {
        timestamp: world.now(),
        actor,
        token_fingerprint: world.details.clone(),
        scope: vec!["read".to_string()],
        expires_at: None,
    };
    world.current_event = Some(event);
}

#[when("I create a TokenRevoked event")]
async fn when_create_token_revoked(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::TokenRevoked {
        timestamp: world.now(),
        actor,
        token_fingerprint: world.details.clone(),
        reason: world.reason.clone(),
    };
    world.current_event = Some(event);
}

#[when("I create a NodeRegistered event")]
async fn when_create_node_registered(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::NodeRegistered {
        timestamp: world.now(),
        actor,
        node_id: world.node_id.clone(),
        gpu_count: 1,
        total_vram_gb: 24,
        capabilities: vec!["cuda".to_string()],
    };
    world.current_event = Some(event);
}

#[when("I create a NodeDeregistered event")]
async fn when_create_node_deregistered(world: &mut BddWorld) {
    let actor = world.create_default_actor();
    let event = AuditEvent::NodeDeregistered {
        timestamp: world.now(),
        actor,
        node_id: world.node_id.clone(),
        reason: world.reason.clone(),
        pools_affected: vec![],
    };
    world.current_event = Some(event);
}

#[when("I create an InferenceExecuted event")]
async fn when_create_inference_executed(world: &mut BddWorld) {
    let event = AuditEvent::InferenceExecuted {
        timestamp: world.now(),
        customer_id: world.customer_id.clone(),
        job_id: world.task_id.clone(),
        model_ref: world.model_ref.clone(),
        tokens_processed: 100,
        provider_id: None,
        result: audit_logging::AuditResult::Success,
    };
    world.current_event = Some(event);
}

#[when("I create a ModelAccessed event")]
async fn when_create_model_accessed(world: &mut BddWorld) {
    let event = AuditEvent::ModelAccessed {
        timestamp: world.now(),
        customer_id: world.customer_id.clone(),
        model_ref: world.model_ref.clone(),
        access_type: world.details.clone(),
        provider_id: None,
    };
    world.current_event = Some(event);
}

#[when("I create a DataDeleted event")]
async fn when_create_data_deleted(world: &mut BddWorld) {
    let event = AuditEvent::DataDeleted {
        timestamp: world.now(),
        customer_id: world.customer_id.clone(),
        data_types: vec!["inference_logs".to_string()],
        reason: world.reason.clone(),
    };
    world.current_event = Some(event);
}

#[when("I create a GdprDataAccessRequest event")]
async fn when_create_gdpr_data_access_request(world: &mut BddWorld) {
    let event = AuditEvent::GdprDataAccessRequest {
        timestamp: world.now(),
        customer_id: world.customer_id.clone(),
        requester: world.user_id.clone(),
        scope: vec!["all".to_string()],
    };
    world.current_event = Some(event);
}

#[when("I create a GdprDataExport event")]
async fn when_create_gdpr_data_export(world: &mut BddWorld) {
    let event = AuditEvent::GdprDataExport {
        timestamp: world.now(),
        customer_id: world.customer_id.clone(),
        data_types: vec!["inference_logs".to_string()],
        export_format: world.details.clone(),
        file_hash: "abc123".to_string(),
    };
    world.current_event = Some(event);
}

#[when("I create a GdprRightToErasure event")]
async fn when_create_gdpr_right_to_erasure(world: &mut BddWorld) {
    let event = AuditEvent::GdprRightToErasure {
        timestamp: world.now(),
        customer_id: world.customer_id.clone(),
        completed_at: world.now(),
        data_types_deleted: vec!["all".to_string()],
    };
    world.current_event = Some(event);
}

#[when("I validate the event")]
async fn when_validate_event(world: &mut BddWorld) {
    if let Some(mut event) = world.current_event.take() {
        let result = audit_logging::validation::validate_event(&mut event);

        match result {
            Ok(()) => {
                world.store_result(Ok(()));
                world.current_event = Some(event);
            }
            Err(e) => {
                world.store_result(Err(e.to_string()));
            }
        }
    } else {
        world.store_result(Err("No event to validate".to_string()));
    }
}

#[when("I serialize the event to JSON")]
async fn when_serialize_event(world: &mut BddWorld) {
    if let Some(event) = &world.current_event {
        match serde_json::to_string(event) {
            Ok(_json) => {
                world.store_result(Ok(()));
            }
            Err(e) => {
                world.store_result(Err(format!("Serialization failed: {}", e)));
            }
        }
    } else {
        world.store_result(Err("No event to serialize".to_string()));
    }
}
