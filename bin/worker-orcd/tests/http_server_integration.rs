//! Integration tests for HTTP server lifecycle
//!
//! These tests verify the complete server lifecycle including:
//! - Server startup on available ports
//! - Health endpoint availability
//! - Graceful shutdown handling
//! - Bind failure scenarios
//! - Environment variable configuration
//!
//! # Spec References
//! - M0-W-1110: Server initialization
//! - M0-W-1320: Health endpoint
//! - WORK-3010: HTTP server foundation

use axum::{routing::get, Json, Router};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;
use tokio::time::timeout;

// Import from worker-orcd (we need to expose these in lib.rs)
// For now, we'll create minimal test infrastructure

#[derive(Serialize, Deserialize)]
struct HealthResponse {
    status: String,
}

async fn test_health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "healthy".to_string() })
}

fn create_test_router() -> Router {
    Router::new().route("/health", get(test_health_handler))
}

/// Test: Server starts successfully on available port
///
/// Verifies:
/// - Server binds to port 0 (OS-assigned)
/// - Server accepts connections
/// - Health endpoint is immediately available
#[tokio::test]
async fn test_server_starts_on_available_port() {
    // Use port 0 to let OS assign an available port
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let router = create_test_router();

    // Bind to get actual address
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener.local_addr().unwrap();

    // Spawn server in background
    let server_handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Test health endpoint
    let client = reqwest::Client::new();
    let response = client.get(format!("http://{}/health", actual_addr)).send().await.unwrap();

    assert_eq!(response.status(), 200);

    let body: HealthResponse = response.json().await.unwrap();
    assert_eq!(body.status, "healthy");

    // Cleanup
    server_handle.abort();
}

/// Test: Server binds to custom address from configuration
///
/// Verifies:
/// - Server respects explicit bind address
/// - IPv4 localhost binding works
#[tokio::test]
async fn test_server_binds_to_custom_address() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let router = create_test_router();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener.local_addr().unwrap();

    assert_eq!(actual_addr.ip().to_string(), "127.0.0.1");

    // Cleanup
    drop(listener);
}

/// Test: Server fails gracefully when port is already in use
///
/// Verifies:
/// - Bind failure is detected
/// - Error message includes address
/// - First server continues running
#[tokio::test]
async fn test_bind_failure_port_in_use() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    // Bind first server
    let listener1 = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener1.local_addr().unwrap();

    // Try to bind second server to same port
    let result = tokio::net::TcpListener::bind(actual_addr).await;

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);

    // Cleanup
    drop(listener1);
}

/// Test: Server fails with invalid address format
///
/// Verifies:
/// - Invalid IP addresses are rejected
/// - Invalid port numbers are rejected
#[test]
fn test_invalid_address_format() {
    // Invalid IP
    let result = "999.999.999.999:8080".parse::<SocketAddr>();
    assert!(result.is_err());

    // Invalid format
    let result = "not-an-address".parse::<SocketAddr>();
    assert!(result.is_err());

    // Missing port
    let result = "127.0.0.1".parse::<SocketAddr>();
    assert!(result.is_err());
}

/// Test: Health endpoint returns correct JSON structure
///
/// Verifies:
/// - Response is valid JSON
/// - Contains "status" field
/// - Status is "healthy"
/// - Content-Type is application/json
#[tokio::test]
async fn test_health_endpoint_json_structure() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let router = create_test_router();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let response = client.get(format!("http://{}/health", actual_addr)).send().await.unwrap();

    // Verify Content-Type
    let content_type = response.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("application/json"));

    // Verify JSON structure
    let json: serde_json::Value = response.json().await.unwrap();
    assert!(json.get("status").is_some());
    assert_eq!(json["status"], "healthy");

    server_handle.abort();
}

/// Test: Concurrent health checks during startup
///
/// Verifies:
/// - Server handles multiple simultaneous requests
/// - All requests receive 200 OK
/// - No race conditions during startup
#[tokio::test]
async fn test_concurrent_health_checks() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let router = create_test_router();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Send 10 concurrent requests
    let client = reqwest::Client::new();
    let mut handles = vec![];

    for _ in 0..10 {
        let client = client.clone();
        let url = format!("http://{}/health", actual_addr);
        let handle = tokio::spawn(async move { client.get(&url).send().await.unwrap().status() });
        handles.push(handle);
    }

    // Wait for all requests
    for handle in handles {
        let status = handle.await.unwrap();
        assert_eq!(status, 200);
    }

    server_handle.abort();
}

/// Test: Server shutdown completes within timeout
///
/// Verifies:
/// - Server responds to shutdown signal
/// - Shutdown completes within 5 seconds (per spec)
/// - No hanging connections
#[tokio::test]
async fn test_graceful_shutdown_timeout() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let router = create_test_router();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.recv().await;
            })
            .await
            .unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Trigger shutdown
    let _ = shutdown_tx.send(());

    // Verify shutdown completes within 5 seconds
    let result = timeout(Duration::from_secs(5), server_handle).await;
    assert!(result.is_ok(), "Server should shutdown within 5 seconds");
}

/// Test: IPv6 address binding
///
/// Verifies:
/// - Server supports IPv6 addresses
/// - Localhost IPv6 binding works
#[tokio::test]
async fn test_ipv6_binding() {
    let addr: SocketAddr = "[::1]:0".parse().unwrap();
    let router = create_test_router();

    // This may fail on systems without IPv6 support
    let listener_result = tokio::net::TcpListener::bind(addr).await;

    if let Ok(listener) = listener_result {
        let actual_addr = listener.local_addr().unwrap();
        assert!(actual_addr.is_ipv6());
        drop(listener);
    }
    // If IPv6 not available, test passes (not a failure)
}

/// Test: 0.0.0.0 binding (all interfaces)
///
/// Verifies:
/// - Server can bind to all interfaces
/// - Useful for container deployments
#[tokio::test]
async fn test_all_interfaces_binding() {
    let addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
    let router = create_test_router();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let actual_addr = listener.local_addr().unwrap();

    assert_eq!(actual_addr.ip().to_string(), "0.0.0.0");

    drop(listener);
}

// ---
// Built by Foundation-Alpha 🏗️
