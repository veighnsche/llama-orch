# keeper-config-contract

Keeper configuration contract for the rbee ecosystem.

## Purpose

This crate provides the configuration schema for rbee-keeper, ensuring stability across versions.

## Configuration

Configuration is loaded from `~/.config/rbee/config.toml`:

```toml
queen_port = 7833
```

## Usage

```rust
use keeper_config_contract::KeeperConfig;

// Create default config
let config = KeeperConfig::default();

// Get queen URL
let url = config.queen_url();
println!("Queen URL: {}", url);

// Validate config
config.validate()?;

// Parse from TOML
let toml_str = r#"
queen_port = 8080
"#;
let config = KeeperConfig::from_toml(toml_str)?;

// Serialize to TOML
let toml_str = config.to_toml()?;
```

## Validation

Configuration is validated to ensure:
- Port is in valid range (1024-65535)
- Port is not privileged (< 1024)

## Add to Cargo.toml

```toml
[dependencies]
keeper-config-contract = { path = "../97_contracts/keeper-config-contract" }
```

## License

GPL-3.0-or-later
