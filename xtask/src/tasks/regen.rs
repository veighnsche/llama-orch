use crate::util::{repo_root, write_if_changed};
use anyhow::{anyhow, Context, Result};
use openapiv3;
use serde_yaml;
use std::{fs, process::Command};

pub fn regen_all() -> Result<()> {
    regen_openapi()?;
    regen_schema()?;
    spec_extract()?;
    println!("regen: OK");
    Ok(())
}

pub fn spec_extract() -> Result<()> {
    let status = Command::new("cargo")
        .arg("run")
        .arg("-p")
        .arg("tools-spec-extract")
        .arg("--quiet")
        .status()
        .context("running tools-spec-extract")?;
    if !status.success() {
        return Err(anyhow!("spec-extract failed"));
    }
    println!("spec-extract: OK");
    Ok(())
}

pub fn regen_openapi() -> Result<()> {
    let root = repo_root()?;
    let data = root.join("contracts/openapi/data.yaml");
    let control = root.join("contracts/openapi/control.yaml");

    // Validate both OpenAPI documents
    let _: openapiv3::OpenAPI = serde_yaml::from_str(&fs::read_to_string(&data)?)
        .with_context(|| format!("parsing {}", data.display()))?;
    let _: openapiv3::OpenAPI = serde_yaml::from_str(&fs::read_to_string(&control)?)
        .with_context(|| format!("parsing {}", control.display()))?;

    // Deterministic generated types for data plane (handcrafted mapping for now)
    let api_types_out = root.join("contracts/api-types/src/generated.rs");
    let api_types_src =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/templates/generated_api_types.rs"));
    write_if_changed(&api_types_out, api_types_src)?;

    // Deterministic generated client for data plane
    let client_out = root.join("tools/openapi-client/src/generated.rs");
    let client_src =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/templates/generated_client.rs"));
    write_if_changed(&client_out, client_src)?;

    // Ensure formatting so dev:loop fmt check will pass
    let _ = Command::new("cargo").arg("fmt").arg("--all").status();

    println!("regen-openapi: OK (validated + generated)");
    Ok(())
}

pub fn regen_schema() -> Result<()> {
    let root = repo_root()?;
    let out = root.join("contracts/schemas/config.schema.json");
    contracts_config_schema::emit_schema_json(&out)
        .map_err(|e| anyhow!("emit schema failed: {e}"))?;
    println!("regen-schema: OK");

    // Ensure formatting so dev:loop fmt check will pass
    let _ = Command::new("cargo").arg("fmt").arg("--all").status();
    Ok(())
}
