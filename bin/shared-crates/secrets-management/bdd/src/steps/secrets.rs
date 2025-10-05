//! Secret loading and verification step definitions

use super::world::BddWorld;
use cucumber::{given, when};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use tempfile::TempDir;

#[given(expr = "a secret file at {string}")]
async fn given_secret_file(world: &mut BddWorld, path: String) {
    world.secret_path = Some(PathBuf::from(path));
}

#[given(expr = "a secret file with permissions {int}")]
async fn given_secret_file_with_permissions(world: &mut BddWorld, mode: u32) {
    world.file_mode = Some(mode);
}

#[given(expr = "a secret file containing {string}")]
async fn given_secret_file_content(world: &mut BddWorld, content: String) {
    world.secret_value = Some(content);
}

#[given(expr = "a secret file containing 2MB of data")]
async fn given_large_secret_file(world: &mut BddWorld) {
    // Create 2MB of data
    let large_content = "a".repeat(2 * 1024 * 1024);
    world.secret_value = Some(large_content);
}

#[given(expr = "CREDENTIALS_DIRECTORY is set to {string}")]
async fn given_credentials_directory(world: &mut BddWorld, path: String) {
    std::env::set_var("CREDENTIALS_DIRECTORY", path);
    world.credentials_dir_set = true;
}

#[given(expr = "a token {string}")]
async fn given_token(world: &mut BddWorld, token: String) {
    world.token = Some(token);
}

#[given(expr = "a domain {string}")]
async fn given_domain(world: &mut BddWorld, domain: String) {
    world.domain = Some(domain.into_bytes());
}

#[given(expr = "a systemd credential {string}")]
async fn given_systemd_credential(world: &mut BddWorld, name: String) {
    world.credential_name = Some(name);
}

#[when("I load the secret from file")]
async fn when_load_secret_from_file(world: &mut BddWorld) {
    // Create a temporary file with the specified content and permissions
    let content = world.secret_value.as_ref().map(|s| s.as_str()).unwrap_or("default-test-secret");

    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("secret");

    fs::write(&file_path, content).unwrap();

    // Set permissions if specified, default to 0600
    let mode = world.file_mode.unwrap_or(0o600);
    let mut perms = fs::metadata(&file_path).unwrap().permissions();
    perms.set_mode(mode);
    fs::set_permissions(&file_path, perms).unwrap();

    // Try to load the secret
    let result = secrets_management::Secret::load_from_file(&file_path);

    match result {
        Ok(secret) => {
            world.secret_loaded = Some(secret);
            world.store_result(Ok(()));
        }
        Err(e) => {
            world.store_result(Err(e));
        }
    }

    // Keep temp_dir alive
    world.temp_dir = Some(temp_dir);
}

#[when(expr = "I verify the secret with {string}")]
async fn when_verify_secret(world: &mut BddWorld, input: String) {
    world.verify_input = Some(input.clone());

    // Use the loaded secret if available
    if let Some(ref secret) = world.secret_loaded {
        world.verify_result = Some(secret.verify(&input));
    } else if let Some(ref secret_value) = world.secret_value {
        // Fallback to direct comparison for testing
        world.verify_result = Some(secret_value == &input);
    }
}

#[when("I derive a key from the token")]
async fn when_derive_key(world: &mut BddWorld) {
    if let (Some(ref token), Some(ref domain)) = (&world.token, &world.domain) {
        let result = secrets_management::SecretKey::derive_from_token(token, domain);

        match result {
            Ok(key) => {
                // Store hex representation
                world.derived_key = Some(hex::encode(key.as_bytes()));
                world.store_result(Ok(()));
            }
            Err(e) => {
                world.store_result(Err(e));
            }
        }
    } else {
        world.store_result(Err(secrets_management::SecretError::InvalidFormat(
            "missing token or domain".to_string(),
        )));
    }
}

#[when("I load from systemd credential")]
async fn when_load_systemd_credential(world: &mut BddWorld) {
    // Systemd credentials require $CREDENTIALS_DIRECTORY to be set
    if let Some(ref cred_name) = world.credential_name {
        // Only create temp directory if CREDENTIALS_DIRECTORY wasn't already set by a Given step
        if !world.credentials_dir_set {
            // Create a temporary credentials directory
            let temp_dir = TempDir::new().unwrap();
            std::env::set_var("CREDENTIALS_DIRECTORY", temp_dir.path());

            // Create the credential file
            let cred_path = temp_dir.path().join(cred_name);
            fs::write(&cred_path, "test-credential-value").unwrap();

            // Set proper permissions
            let mut perms = fs::metadata(&cred_path).unwrap().permissions();
            perms.set_mode(0o600);
            fs::set_permissions(&cred_path, perms).unwrap();

            // Keep temp_dir alive
            world.temp_dir = Some(temp_dir);
        }

        // Try to load (will use either the Given-set path or our temp path)
        let result = secrets_management::Secret::from_systemd_credential(cred_name);

        match result {
            Ok(secret) => {
                world.secret_loaded = Some(secret);
                world.store_result(Ok(()));
            }
            Err(e) => {
                world.store_result(Err(e));
            }
        }
    } else {
        world.store_result(Err(secrets_management::SecretError::InvalidFormat(
            "no credential name".to_string(),
        )));
    }
}
