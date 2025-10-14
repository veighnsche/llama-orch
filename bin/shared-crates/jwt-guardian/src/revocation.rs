//! Redis-backed JWT revocation list

use crate::{JwtError, Result};
use redis::{aio::MultiplexedConnection, AsyncCommands, Client};

/// Redis-backed revocation list for JWT tokens
///
/// Stores revoked JWT IDs (jti) with TTL matching token expiration.
pub struct RevocationList {
    conn: MultiplexedConnection,
}

impl RevocationList {
    /// Connect to Redis
    ///
    /// # Arguments
    ///
    /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379")
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use jwt_guardian::RevocationList;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let revocation = RevocationList::connect("redis://localhost:6379").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn connect(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)
            .map_err(|e| JwtError::RedisError(format!("Failed to create client: {}", e)))?;

        let conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| JwtError::RedisError(format!("Failed to connect: {}", e)))?;

        Ok(Self { conn })
    }

    /// Check if token is revoked
    ///
    /// # Arguments
    ///
    /// * `jti` - JWT ID to check
    ///
    /// # Returns
    ///
    /// `true` if token is revoked, `false` otherwise
    pub async fn is_revoked(&mut self, jti: &str) -> Result<bool> {
        let key = format!("jwt:revoked:{}", jti);
        let exists: bool = self.conn.exists(&key).await?;
        Ok(exists)
    }

    /// Revoke token
    ///
    /// # Arguments
    ///
    /// * `jti` - JWT ID to revoke
    /// * `exp` - Token expiration time (Unix timestamp)
    ///
    /// Sets TTL to match token expiration (automatic cleanup)
    pub async fn revoke(&mut self, jti: &str, exp: i64) -> Result<()> {
        let key = format!("jwt:revoked:{}", jti);
        let now = chrono::Utc::now().timestamp();
        let ttl = (exp - now).max(0) as u64;

        // Set key with TTL
        self.conn.set_ex(&key, 1, ttl).await?;

        Ok(())
    }

    /// Clear expired revocations (maintenance)
    ///
    /// Redis automatically removes expired keys, but this can be used
    /// for manual cleanup or monitoring.
    ///
    /// # Returns
    ///
    /// Number of keys cleaned up
    pub async fn cleanup_expired(&mut self) -> Result<usize> {
        // Redis handles TTL automatically, this is a no-op
        // Kept for API compatibility and future manual cleanup
        Ok(0)
    }

    /// Get count of revoked tokens (for monitoring)
    pub async fn count_revoked(&mut self) -> Result<usize> {
        let pattern = "jwt:revoked:*";
        let keys: Vec<String> = self.conn.keys(pattern).await?;
        Ok(keys.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Integration tests require Redis running
    // These are unit tests for the structure

    #[tokio::test]
    #[ignore] // Requires Redis
    async fn test_revocation_list_connect() {
        let result = RevocationList::connect("redis://localhost:6379").await;
        // Will fail if Redis not running - that's expected
        let _ = result;
    }
}
