// Real llama.cpp E2E (CPU) — ignored by default.
// Requirements to run:
// - Env LLORCH_E2E_REAL=1
// - Env LLORCH_E2E_MODEL_PATH=/absolute/path/to/valid/model.gguf (small GGUF)
// - Tools in PATH (when running on host): git, cmake, make, gcc, g++, pkg-config, and libcurl dev headers (via pkg-config: libcurl)
// - No package installs will be attempted (allow_package_installs=false); preflight will error with actionable hints if missing
//
// Usage:
//   LLORCH_E2E_REAL=1 \
//   LLORCH_E2E_MODEL_PATH=/models/TinyLLama.gguf \
//   LLAMA_REF=${LLAMA_REF:-master} \
//   cargo test -p provisioners-engine-provisioner --test llamacpp_source_cpu_real_e2e -- --ignored --nocapture

#![cfg(unix)]

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use provisioners_engine_provisioner::{provider_for, EngineProvisioner};

#[test]
#[ignore]
fn llamacpp_source_cpu_real_e2e() -> anyhow::Result<()> {
    if std::env::var("LLORCH_E2E_REAL").as_deref() != Ok("1") {
        eprintln!("skipping real llama.cpp E2E (set LLORCH_E2E_REAL=1)");
        return Ok(());
    }
    let model_path = match std::env::var("LLORCH_E2E_MODEL_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("skipping: LLORCH_E2E_MODEL_PATH not set to a valid GGUF file");
            return Ok(());
        }
    };
    if !model_path.exists() {
        eprintln!("skipping: model path {:?} does not exist", model_path);
        return Ok(());
    }
    // default to master unless overridden
    let llama_ref = std::env::var("LLAMA_REF").unwrap_or_else(|_| "master".to_string());

    let tmp = tempfile::tempdir()?;
    let cache_dir = tmp.path().join("cache");

    let mut pool = provisioners_engine_provisioner::cfg::PoolConfig {
        id: "p-real".into(),
        engine: provisioners_engine_provisioner::cfg::Engine::Llamacpp,
        model: "m-real".into(),
        quant: None,
        ctx: None,
        devices: vec![0],
        tensor_split: None,
        preload: None,
        require_same_engine_version: None,
        sampler_profile_version: None,
        provisioning: provisioners_engine_provisioner::cfg::ProvisioningConfig::default(),
        queue: provisioners_engine_provisioner::cfg::QueueConfig::default(),
        admission: provisioners_engine_provisioner::cfg::AdmissionConfig::default(),
        timeouts: provisioners_engine_provisioner::cfg::Timeouts::default(),
    };
    pool.provisioning.allow_package_installs = Some(false);
    pool.provisioning.source = Some(provisioners_engine_provisioner::cfg::SourceConfig {
        repo: "https://github.com/ggml-org/llama.cpp".to_string(),
        r#ref: llama_ref,
        submodules: Some(false),
        build: provisioners_engine_provisioner::cfg::SourceBuildConfig {
            cmake_flags: Some(vec![]), // CPU build
            generator: None,
            cache_dir: Some(cache_dir.to_string_lossy().to_string()),
        },
    });
    pool.provisioning.model.r#ref = Some(model_path.to_string_lossy().to_string());
    pool.provisioning.ports = Some(vec![0]);

    let prov = provider_for(&pool)?;
    prov.ensure(&pool)?;

    // Validate handoff and health
    let handoff_path = PathBuf::from(".runtime").join("engines").join("llamacpp.json");
    let handoff = fs::read_to_string(&handoff_path)?;
    let v: serde_json::Value = serde_json::from_str(&handoff)?;
    assert_eq!(v["engine"], "llamacpp");
    let url = v["url"].as_str().unwrap();
    assert!(url.starts_with("http://127.0.0.1:"));
    let port: u16 = url.strip_prefix("http://127.0.0.1:").unwrap().parse().unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let mut ok = false;
    while std::time::Instant::now() < deadline {
        if provisioners_engine_provisioner::util::http_ok("127.0.0.1", port, "/health").unwrap_or(false) {
            ok = true; break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    assert!(ok, "llama-server did not become healthy");

    // After health is OK: request a haiku that includes the current minute spelled out.
    // 1) Build minute word and prompt
    let minute_word = current_minute_word();
    let prompt = format!(
        "Write a three-line haiku that contains the exact word '{mw}' exactly once. Only output the three haiku lines, without any extra commentary.",
        mw = minute_word
    );

    // 2) Try OpenAI-like /v1/completions first, then fallback to native /completion
    let body_v1 = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 80,
        "temperature": 0.7
    })
    .to_string();
    let resp1 = http_post("127.0.0.1", port, "/v1/completions", &body_v1).ok();
    let text = resp1
        .as_deref()
        .and_then(parse_text_from_llamacpp)
        .or_else(|| {
            let body_native = serde_json::json!({
                "prompt": prompt,
                "n_predict": 80,
                "temperature": 0.7
            })
            .to_string();
            http_post("127.0.0.1", port, "/completion", &body_native)
                .ok()
                .as_deref()
                .and_then(parse_text_from_llamacpp)
        })
        .unwrap_or_else(|| "<no text returned>".to_string());

    // 3) Print the HAIKU clearly in terminal
    println!("\n======== HAIKU ========\n\n{}\n\n======== HAIKU ========\n", text.trim());

    // Cleanup pid
    let pid_path = provisioners_engine_provisioner::util::default_run_dir().join("p-real.pid");
    if let Ok(s) = fs::read_to_string(&pid_path) {
        let pid = s.trim();
        let _ = std::process::Command::new("/bin/kill").arg(pid).status();
        let _ = fs::remove_file(&pid_path);
    }

    Ok(())
}

fn current_minute_word() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let minute = ((secs / 60) % 60) as u8;
    minute_to_words(minute)
}

fn minute_to_words(m: u8) -> String {
    const ONES: [&str; 20] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ];
    const TENS: [&str; 6] = ["", "", "twenty", "thirty", "forty", "fifty"];
    if m < 20 { return ONES[m as usize].to_string(); }
    let tens = (m / 10) as usize; // 2..5
    let ones = (m % 10) as usize;
    if ones == 0 { TENS[tens].to_string() } else { format!("{}-{}", TENS[tens], ONES[ones]) }
}

fn http_post(host: &str, port: u16, path: &str, body: &str) -> anyhow::Result<String> {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(addr)?;
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path, host, body.len(), body
    );
    stream.write_all(req.as_bytes())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    let resp = String::from_utf8_lossy(&buf).into_owned();
    // Split headers/body
    if let Some(idx) = resp.find("\r\n\r\n") {
        Ok(resp[idx + 4..].to_string())
    } else {
        Ok(resp)
    }
}

fn parse_text_from_llamacpp(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    // OpenAI-like completions
    if let Some(s) = v.get("choices").and_then(|c| c.get(0)).and_then(|c0| c0.get("text")).and_then(|t| t.as_str()) {
        return Some(s.to_string());
    }
    // chat.completions style
    if let Some(s) = v.get("choices").and_then(|c| c.get(0)).and_then(|c0| c0.get("message")).and_then(|m| m.get("content")).and_then(|t| t.as_str()) {
        return Some(s.to_string());
    }
    // native llama.cpp server variants seen in the wild
    if let Some(s) = v.get("content").and_then(|t| t.as_str()) { return Some(s.to_string()); }
    if let Some(s) = v.get("completion").and_then(|t| t.as_str()) { return Some(s.to_string()); }
    None
}
