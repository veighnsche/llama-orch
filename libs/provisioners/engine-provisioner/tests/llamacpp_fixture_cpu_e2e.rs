// E2E-like hermetic test: build and run a tiny fixture "llama-server" via CMake
// and exercise the LlamaCppSourceProvisioner end-to-end in CPU mode.
//
// - Hermetic: no network fetch of models, no package installs.
// - The git repo is created on-the-fly in a tempdir, then cloned via file:// path.
// - The built server responds to /health and /version.
// - Ignored by default; run with: LLORCH_E2E_FIXTURE=1 cargo test -p provisioners-engine-provisioner --test llamacpp_fixture_cpu_e2e -- --ignored --nocapture

#![cfg(unix)]

use std::fs;
use std::io::Write;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use provisioners_engine_provisioner::{provider_for, EngineProvisioner};

fn http_health_ok(host: &str, port: u16) -> bool {
    let addr = format!("{}:{}", host, port);
    if let Ok(mut stream) = TcpStream::connect(addr) {
        let req = format!("GET /health HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", host);
        let _ = stream.write_all(req.as_bytes());
        let mut buf = String::new();
        let _ = std::io::Read::read_to_string(&mut stream, &mut buf);
        return buf.starts_with("HTTP/1.1 200") || buf.starts_with("HTTP/1.0 200");
    }
    false
}

fn write_fixture_repo(dir: &std::path::Path) {
    fs::create_dir_all(dir).unwrap();
    // Minimal C HTTP server that answers 200 on /health and returns a JSON version on /version
    let server_c = r#"#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int starts_with(const char* s, const char* p) { return strncmp(s, p, strlen(p)) == 0; }

int main(int argc, char** argv) {
    const char* host = "127.0.0.1";
    int port = 8080;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--host") == 0) host = argv[i+1];
        if (strcmp(argv[i], "--port") == 0) port = atoi(argv[i+1]);
    }

    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); return 1; }
    int opt = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(host);
    if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(s, 16) < 0) { perror("listen"); return 1; }

    while (1) {
        int c = accept(s, NULL, NULL);
        if (c < 0) continue;
        char buf[2048]; int n = read(c, buf, sizeof(buf)-1);
        if (n <= 0) { close(c); continue; }
        buf[n] = '\0';
        if (starts_with(buf, "GET /health")) {
            const char* resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
            write(c, resp, strlen(resp));
        } else if (starts_with(buf, "GET /version")) {
            const char* body = "{\"version\":\"fixture-1.0\"}";
            char hdr[256];
            snprintf(hdr, sizeof(hdr), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", strlen(body));
            write(c, hdr, strlen(hdr));
            write(c, body, strlen(body));
        } else {
            const char* resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            write(c, resp, strlen(resp));
        }
        close(c);
    }
    close(s);
    return 0;
}
"#;
    let cmakelists = r#"cmake_minimum_required(VERSION 3.16)
project(llama_fixture C)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(llama-server server.c)
"#;
    fs::write(dir.join("server.c"), server_c).unwrap();
    fs::write(dir.join("CMakeLists.txt"), cmakelists).unwrap();

    // Initialize git repo and tag
    let status = Command::new("git").current_dir(dir).arg("init").status().unwrap();
    assert!(status.success());
    let status = Command::new("git").current_dir(dir).args(["add", "."]).status().unwrap();
    assert!(status.success());
    let status = Command::new("git").current_dir(dir).args(["commit", "-m", "fixture"]).status().unwrap();
    assert!(status.success());
    let status = Command::new("git").current_dir(dir).args(["tag", "vtest"]).status().unwrap();
    assert!(status.success());
}

#[test]
#[ignore]
fn llamacpp_fixture_cpu_e2e() -> anyhow::Result<()> {
    if std::env::var("LLORCH_E2E_FIXTURE").as_deref() != Ok("1") {
        eprintln!("skipping fixture E2E (set LLORCH_E2E_FIXTURE=1 to run)");
        return Ok(());
    }

    // 1) Create a hermetic fixture repo
    let tmp = tempfile::tempdir()?;
    let repo = tmp.path().join("repo");
    write_fixture_repo(&repo);

    // 2) Create a dummy model file
    let model_path = tmp.path().join("test.gguf");
    fs::write(&model_path, b"gguf-dummy")?;

    // 3) Build a pool config targeting the fixture repo via local path
    let mut pool = provisioners_engine_provisioner::cfg::PoolConfig {
        id: "p-fixture".into(),
        engine: provisioners_engine_provisioner::cfg::Engine::Llamacpp,
        model: "m1".into(),
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
        repo: repo.to_string_lossy().to_string(), // local path clone
        r#ref: "vtest".into(),
        submodules: Some(false),
        build: provisioners_engine_provisioner::cfg::SourceBuildConfig {
            cmake_flags: Some(vec![]),
            generator: None,
            cache_dir: Some(tmp.path().join("cache").to_string_lossy().to_string()),
        },
    });
    pool.provisioning.model.r#ref = Some(model_path.to_string_lossy().to_string());
    pool.provisioning.ports = Some(vec![0]); // let provisioner pick a free port

    // 4) Run ensure()
    let prov = provider_for(&pool)?;
    prov.ensure(&pool)?;

    // 5) Validate handoff file (contract shape + model staging) and server health
    let handoff_path = PathBuf::from(".runtime").join("engines").join("llamacpp.json");
    let handoff = fs::read_to_string(&handoff_path)?;
    let v: serde_json::Value = serde_json::from_str(&handoff)?;
    // Contract shape for downstream consumers
    assert_eq!(v["engine"], "llamacpp");
    assert!(v.get("engine_version").is_some());
    assert!(v.get("provisioning_mode").is_some());
    assert!(v.get("pool_id").is_some());
    assert!(v.get("flags").is_some());
    assert!(v.get("model").is_some());
    let url = v["url"].as_str().unwrap_or("");
    assert!(url.starts_with("http://127.0.0.1:"));
    let port: u16 = url.strip_prefix("http://127.0.0.1:").unwrap().parse().unwrap();

    // Model staging: the handoff must point to a readable model path that exists
    let model_obj = v.get("model").unwrap();
    let model_path_str = model_obj.get("path").and_then(|p| p.as_str()).unwrap();
    assert_eq!(model_path_str, model_path.to_string_lossy());
    let meta = fs::metadata(&model_path_str)?;
    assert!(meta.is_file());
    // Downstream crates would open the model; simulate a simple read open
    let _f = fs::File::open(&model_path_str)?;

    // ensure server answers /health (consumer-ready)
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut ok = false;
    while std::time::Instant::now() < deadline {
        if http_health_ok("127.0.0.1", port) { ok = true; break; }
        std::thread::sleep(Duration::from_millis(200));
    }
    assert!(ok, "fixture server did not respond healthy");

    // 6) Cleanup: kill via pid file
    let pid_path = provisioners_engine_provisioner::util::default_run_dir().join("p-fixture.pid");
    if let Ok(s) = fs::read_to_string(&pid_path) {
        let pid = s.trim();
        let _ = Command::new("/bin/kill").arg(pid).status();
        let _ = fs::remove_file(&pid_path);
    }

    Ok(())
}
