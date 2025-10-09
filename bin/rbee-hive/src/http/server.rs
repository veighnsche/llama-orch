//! HTTP server initialization and lifecycle management
//!
//! Mirrors the pattern from llm-worker-rbee/src/http/server.rs
//! 
//! Created by: TEAM-026

use axum::Router;
use std::net::SocketAddr;
use thiserror::Error;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

/// HTTP server errors
#[derive(Debug, Error)]
pub enum ServerError {
    /// Failed to bind to the specified address
    #[error("Failed to bind to {addr}: {source}")]
    BindFailed {
        addr: SocketAddr,
        source: std::io::Error,
    },

    /// Server runtime error
    #[error("Server runtime error: {0}")]
    Runtime(String),

    /// Shutdown error
    #[error("Shutdown error: {0}")]
    Shutdown(String),
}

/// HTTP server with lifecycle management
pub struct HttpServer {
    /// Bind address
    addr: SocketAddr,

    /// Router with all endpoints
    router: Router,

    /// Shutdown signal sender
    shutdown_tx: broadcast::Sender<()>,
}

impl HttpServer {
    /// Create new HTTP server bound to address
    ///
    /// # Arguments
    /// * `addr` - Socket address to bind to (e.g., `0.0.0.0:8080`)
    /// * `router` - Axum router with all endpoints configured
    ///
    /// # Returns
    /// * `Ok(HttpServer)` - Server ready to run
    /// * `Err(ServerError::BindFailed)` - Failed to bind to address
    pub async fn new(addr: SocketAddr, router: Router) -> Result<Self, ServerError> {
        // Create shutdown channel (capacity 1 is sufficient for shutdown signal)
        let (shutdown_tx, _) = broadcast::channel(1);

        info!(
            addr = %addr,
            "rbee-hive HTTP server initialized"
        );

        Ok(Self {
            addr,
            router,
            shutdown_tx,
        })
    }

    /// Run server until shutdown signal received
    ///
    /// This method blocks until:
    /// - `shutdown()` is called
    /// - SIGTERM/SIGINT is received
    ///
    /// The server will complete in-flight requests before shutting down.
    ///
    /// # Returns
    /// * `Ok(())` - Server shut down gracefully
    /// * `Err(ServerError)` - Server encountered an error
    pub async fn run(self) -> Result<(), ServerError> {
        let listener = tokio::net::TcpListener::bind(self.addr)
            .await
            .map_err(|source| {
                error!(
                    addr = %self.addr,
                    error = %source,
                    "Failed to bind to address"
                );
                ServerError::BindFailed {
                    addr: self.addr,
                    source,
                }
            })?;

        info!(
            addr = %self.addr,
            "rbee-hive HTTP server listening"
        );

        // Clone shutdown receiver for signal handler
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        // Spawn signal handler
        let shutdown_tx_clone = self.shutdown_tx.clone();
        tokio::spawn(async move {
            match tokio::signal::ctrl_c().await {
                Ok(()) => {
                    info!("Received SIGINT/SIGTERM, initiating graceful shutdown");
                    let _ = shutdown_tx_clone.send(());
                }
                Err(e) => {
                    error!(error = %e, "Failed to listen for shutdown signal");
                }
            }
        });

        // Run server with graceful shutdown
        axum::serve(listener, self.router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.recv().await;
                warn!("rbee-hive HTTP server shutting down gracefully");
            })
            .await
            .map_err(|e| ServerError::Runtime(e.to_string()))?;

        info!("rbee-hive HTTP server shutdown complete");

        Ok(())
    }

    /// Trigger graceful shutdown
    ///
    /// This method sends a shutdown signal to the running server.
    /// The server will complete in-flight requests before shutting down.
    ///
    /// # Returns
    /// * `Ok(())` - Shutdown signal sent successfully
    /// * `Err(ServerError::Shutdown)` - Failed to send shutdown signal
    pub fn shutdown(&self) -> Result<(), ServerError> {
        self.shutdown_tx
            .send(())
            .map(|_| ())
            .map_err(|e| ServerError::Shutdown(format!("No receivers: {e}")))
    }

    /// Get the bind address
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Json};
    use serde::Serialize;

    #[derive(Serialize)]
    struct TestResponse {
        status: String,
    }

    async fn test_handler() -> Json<TestResponse> {
        Json(TestResponse {
            status: "ok".to_string(),
        })
    }

    #[tokio::test]
    async fn test_server_creation() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let router = Router::new().route("/test", get(test_handler));

        let server = HttpServer::new(addr, router).await;
        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_server_error_display() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let io_error = std::io::Error::new(std::io::ErrorKind::AddrInUse, "port in use");

        let error = ServerError::BindFailed {
            addr,
            source: io_error,
        };

        let error_msg = error.to_string();
        assert!(error_msg.contains("127.0.0.1:8080"));
        assert!(error_msg.contains("Failed to bind"));
    }
}
