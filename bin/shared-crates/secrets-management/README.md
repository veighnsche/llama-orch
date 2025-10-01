# secrets-management

**Secure credential loading and management**

Loads API tokens and secrets from secure sources (files, systemd credentials, secret managers). Never exposes secrets in environment variables or process listings.

**Key responsibilities:**
- Load tokens from files (not environment)
- Support systemd LoadCredential
- Optional: Vault/AWS Secrets Manager integration
- Prevent secrets in logs/metrics
- Secure memory handling (zero on drop)