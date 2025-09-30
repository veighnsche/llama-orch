use anyhow::{anyhow, Context, Result};
use std::process::Command;

pub fn ci_auth_min() -> Result<()> {
    // Minimal self-check for auth-min crate
    let ok1 = auth_min::parse_bearer(Some("Bearer abc")) == Some("abc".to_string());
    let ok2 = !auth_min::timing_safe_eq(b"a", b"b");
    let fp = auth_min::token_fp6("secret");
    let ok3 = fp.len() == 6 && fp.chars().all(|c| c.is_ascii_alphanumeric());
    if ok1 && ok2 && ok3 {
        println!("ci:auth OK (fp6=****{})", fp);
        Ok(())
    } else {
        Err(anyhow!("ci:auth failed"))
    }
}

pub fn docs_index() -> Result<()> {
    let status = Command::new("cargo")
        .arg("run")
        .arg("-p")
        .arg("tools-readme-index")
        .arg("--quiet")
        .status()
        .context("running tools-readme-index")?;
    if !status.success() {
        return Err(anyhow!("docs:index failed"));
    }
    println!("docs:index OK");
    Ok(())
}

pub fn ci_haiku_cpu() -> Result<()> {
    let status = Command::new("cargo")
        .arg("test")
        .arg("-p")
        .arg("test-harness-e2e-haiku")
        .arg("--")
        .arg("--nocapture")
        .status()
        .context("running e2e haiku tests")?;
    if !status.success() {
        return Err(anyhow!("ci:haiku:cpu failed"));
    }
    println!("ci:haiku:cpu OK");
    Ok(())
}

pub fn ci_determinism() -> Result<()> {
    let status = Command::new("cargo")
        .arg("test")
        .arg("-p")
        .arg("test-harness-determinism-suite")
        .arg("--")
        .arg("--nocapture")
        .status()
        .context("running determinism suite")?;
    if !status.success() {
        return Err(anyhow!("ci:determinism failed"));
    }
    println!("ci:determinism OK");
    Ok(())
}

pub fn pact_verify() -> Result<()> {
    let status = Command::new("cargo")
        .arg("test")
        .arg("-p")
        .arg("orchestratord")
        .arg("--test")
        .arg("provider_verify")
        .arg("--")
        .arg("--nocapture")
        .status()
        .context("running pact provider verification")?;
    if !status.success() {
        return Err(anyhow!("pact:verify failed"));
    }
    println!("pact:verify OK");
    Ok(())
}

pub fn dev_loop() -> Result<()> {
    // 1) Formatting
    let status = Command::new("cargo")
        .arg("fmt")
        .arg("--all")
        .arg("--")
        .arg("--check")
        .status()
        .context("running cargo fmt")?;
    if !status.success() {
        return Err(anyhow!("fmt failed"));
    }

    // 2) Clippy
    let status = Command::new("cargo")
        .arg("clippy")
        .arg("--all-targets")
        .arg("--all-features")
        .arg("--")
        .arg("-D")
        .arg("warnings")
        .status()
        .context("running cargo clippy")?;
    if !status.success() {
        return Err(anyhow!("clippy failed"));
    }

    // 3) Regenerators
    crate::tasks::regen::regen_openapi()?;
    crate::tasks::regen::regen_schema()?;
    crate::tasks::regen::spec_extract()?;

    // 4) Tests (workspace)
    let status = Command::new("cargo")
        .arg("test")
        .arg("--workspace")
        .arg("--all-features")
        .arg("--")
        .arg("--nocapture")
        .status()
        .context("running workspace tests")?;
    if !status.success() {
        return Err(anyhow!("workspace tests failed"));
    }

    // 5) Link checker
    let status = Command::new("bash")
        .arg("ci/scripts/check_links.sh")
        .status()
        .context("running link checker")?;
    if !status.success() {
        return Err(anyhow!("link check failed"));
    }

    println!("dev:loop OK");
    Ok(())
}
