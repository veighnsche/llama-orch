use anyhow::{Context, Result};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Configuration for worker isolation test
pub struct WorkerTestConfig {
    pub worker_id: String,
    pub model_path: PathBuf,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub port: u16,
    pub hive_port: u16,
    pub timeout_secs: u64,
}

impl Default for WorkerTestConfig {
    fn default() -> Self {
        // Use higher port numbers to avoid conflicts
        Self {
            worker_id: format!("test-worker-{}", uuid::Uuid::new_v4()),
            model_path: PathBuf::from(
                ".test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            ),
            model_ref: "hf:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            port: 28081,      // Higher port to avoid conflicts
            hive_port: 29200, // Higher port to avoid conflicts
            timeout_secs: 30,
        }
    }
}

/// Heartbeat data received from worker (matches rbee_heartbeat crate)
#[derive(Debug, Clone, serde::Deserialize)]
struct HeartbeatPayload {
    worker_id: String,
    timestamp: String,
    health_status: String,
}

/// Shared state for mock hive server
#[derive(Clone)]
struct MockHiveState {
    heartbeats: Arc<Mutex<Vec<HeartbeatPayload>>>,
}

impl MockHiveState {
    fn new() -> Self {
        Self { heartbeats: Arc::new(Mutex::new(Vec::new())) }
    }

    fn add_heartbeat(&self, payload: HeartbeatPayload) {
        if let Ok(mut beats) = self.heartbeats.lock() {
            beats.push(payload);
        }
    }

    fn heartbeat_count(&self) -> usize {
        self.heartbeats.lock().map(|b| b.len()).unwrap_or(0)
    }

    fn get_heartbeats(&self) -> Vec<HeartbeatPayload> {
        self.heartbeats.lock().map(|b| b.clone()).unwrap_or_default()
    }
}

/// Mock hive server for testing (Rust implementation)
struct MockHiveServer {
    handle: Option<thread::JoinHandle<()>>,
    port: u16,
    state: MockHiveState,
    shutdown_tx: Option<std::sync::mpsc::Sender<()>>,
}

impl MockHiveServer {
    fn start(port: u16) -> Result<Self> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // Try to bind to port
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
            .with_context(|| format!("Failed to bind to port {}", port))?;

        // Set non-blocking so we can check shutdown signal
        listener.set_nonblocking(true).context("Failed to set listener to non-blocking")?;

        println!("✓ Mock hive server started on port {}", port);

        let state = MockHiveState::new();
        let state_clone = state.clone();

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel();

        // Spawn server thread
        let handle = thread::spawn(move || {
            loop {
                // Check for shutdown signal
                if shutdown_rx.try_recv().is_ok() {
                    break;
                }

                // Try to accept connection (non-blocking)
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let state = state_clone.clone();

                        // Read HTTP request
                        let mut buffer = [0; 4096];
                        if let Ok(n) = stream.read(&mut buffer) {
                            let request = String::from_utf8_lossy(&buffer[..n]);

                            // Parse request - heartbeat sends to /v1/heartbeat
                            if request.contains("POST /v1/heartbeat") {
                                // Find JSON body
                                if let Some(body_start) = request.find("\r\n\r\n") {
                                    let body = &request[body_start + 4..];

                                    // Try to parse JSON
                                    if let Ok(payload) =
                                        serde_json::from_str::<HeartbeatPayload>(body)
                                    {
                                        let count = state.heartbeat_count() + 1;
                                        let timestamp = chrono::Local::now().format("%H:%M:%S");

                                        eprintln!(
                                            "[{}] ✅ Heartbeat #{} from worker: {}",
                                            timestamp, count, payload.worker_id
                                        );
                                        eprintln!(
                                            "           Health: {} at {}",
                                            payload.health_status, payload.timestamp
                                        );

                                        state.add_heartbeat(payload);

                                        // Send 200 OK response
                                        let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}";
                                        let _ = stream.write_all(response.as_bytes());
                                    } else {
                                        // Bad request
                                        let response = "HTTP/1.1 400 Bad Request\r\n\r\n";
                                        let _ = stream.write_all(response.as_bytes());
                                    }
                                }
                            } else {
                                // 404 for other paths
                                let response = "HTTP/1.1 404 Not Found\r\n\r\n";
                                let _ = stream.write_all(response.as_bytes());
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // No connection available, sleep briefly
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self { handle: Some(handle), port, state, shutdown_tx: Some(shutdown_tx) })
    }

    fn heartbeat_count(&self) -> usize {
        self.state.heartbeat_count()
    }

    fn print_logs(&self) -> Result<()> {
        println!("\n📋 Hive server logs:");
        let heartbeats = self.state.get_heartbeats();

        if heartbeats.is_empty() {
            println!("   No heartbeats received");
        } else {
            for (i, hb) in heartbeats.iter().enumerate() {
                println!(
                    "   Heartbeat #{}: worker={}, health={}, time={}",
                    i + 1,
                    hb.worker_id,
                    hb.health_status,
                    hb.timestamp
                );
            }
        }

        Ok(())
    }
}

impl Drop for MockHiveServer {
    fn drop(&mut self) {
        // Send shutdown signal
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        // Wait for thread to finish
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Worker process wrapper
struct WorkerProcess {
    process: Child,
    port: u16,
}

impl WorkerProcess {
    fn start(config: &WorkerTestConfig, hive_port: u16) -> Result<Self> {
        println!("\n🚀 Starting worker...");
        println!("   Worker ID: {}", config.worker_id);
        println!("   Model: {}", config.model_path.display());
        println!("   Backend: {}", config.backend);
        println!("   Port: {}", config.port);
        println!("   Hive URL: http://127.0.0.1:{}", hive_port);

        let process = Command::new("cargo")
            .args(&["run", "--release", "--bin", "llm-worker-rbee", "--"])
            .arg("--worker-id")
            .arg(&config.worker_id)
            .arg("--model")
            .arg(&config.model_path)
            .arg("--model-ref")
            .arg(&config.model_ref)
            .arg("--backend")
            .arg(&config.backend)
            .arg("--device")
            .arg(config.device.to_string())
            .arg("--port")
            .arg(config.port.to_string())
            .arg("--hive-url")
            .arg(format!("http://127.0.0.1:{}", hive_port))
            .arg("--local-mode") // Use local mode for testing (no auth, 127.0.0.1 only)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to start worker process")?;

        println!("✓ Worker started (PID: {})", process.id());

        Ok(Self { process, port: config.port })
    }

    fn wait_for_http_server(&self, timeout_secs: u64) -> Result<()> {
        println!("\n⏳ Waiting for worker HTTP server to start (timeout: {}s)...", timeout_secs);

        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(timeout_secs);

        while start.elapsed() < timeout {
            // Just check if HTTP server is up (health endpoint is public, no auth)
            let url = format!("http://127.0.0.1:{}/health", self.port);
            if let Ok(_) = ureq::get(&url).timeout(Duration::from_secs(2)).call() {
                println!("✅ Worker HTTP server is up!");
                return Ok(());
            }
            thread::sleep(Duration::from_millis(500));
        }

        anyhow::bail!("Worker HTTP server did not start within {}s", timeout_secs)
    }

    fn print_logs(&mut self, lines: usize) -> Result<()> {
        println!("\n📋 Worker logs (last {} lines):", lines);

        if let Some(stderr) = self.process.stderr.take() {
            let reader = BufReader::new(stderr);
            let all_lines: Vec<_> = reader.lines().filter_map(|l| l.ok()).collect();
            let start = all_lines.len().saturating_sub(lines);

            for line in &all_lines[start..] {
                println!("   {}", line);
            }
        }

        Ok(())
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

/// Run worker isolation test
pub fn test_worker_isolation(config: Option<WorkerTestConfig>) -> Result<()> {
    let config = config.unwrap_or_default();

    println!("🧪 Worker Isolation Test");
    println!("==================================\n");

    // Step 1: Check model exists
    println!("📦 Checking model file...");
    if !config.model_path.exists() {
        println!("⚠️  Model not found: {}", config.model_path.display());
        println!("   Skipping test - no model available");
        println!("   To run this test, provide a model path:");
        println!("   cargo xtask worker:test --model /path/to/model.gguf");
        return Ok(());
    }
    println!("✓ Model found: {}", config.model_path.display());

    // Step 2: Start mock hive server
    let hive_server = MockHiveServer::start(config.hive_port)?;
    let actual_hive_port = hive_server.port;

    // Give server time to fully initialize
    thread::sleep(Duration::from_millis(500));

    // Step 3: Start worker (use actual hive port, not config)
    let mut worker = WorkerProcess::start(&config, actual_hive_port)?;

    // Step 4: Wait for worker HTTP server to start
    if let Err(e) = worker.wait_for_http_server(config.timeout_secs) {
        println!("\n❌ Worker HTTP server failed to start: {}", e);
        let _ = worker.print_logs(50);
        let _ = hive_server.print_logs();
        return Err(e);
    }

    // Step 5: Wait for first heartbeat (worker sends every 30s, max wait 35s)
    println!("\n⏳ Waiting for first heartbeat (max 35 seconds)...");

    // Poll for heartbeat - STOP as soon as we get one!
    let start = std::time::Instant::now();
    let mut heartbeat_count = 0;
    while start.elapsed() < Duration::from_secs(35) {
        thread::sleep(Duration::from_secs(1));
        heartbeat_count = hive_server.heartbeat_count();
        if heartbeat_count > 0 {
            println!("✅ Received first heartbeat after {:.1}s!", start.elapsed().as_secs_f32());
            break;
        }
    }

    // Step 6: Check heartbeat (just to verify worker is ready)
    if heartbeat_count == 0 {
        println!("\n❌ No heartbeat received - worker not ready");
        let _ = worker.process.kill();
        let _ = worker.process.wait();
        anyhow::bail!("Worker failed to send heartbeat")
    }
    println!("✅ Worker is ready (heartbeat received)");

    // Step 7: Test inference with dual-call pattern
    println!("\n🤔 Testing inference with dual-call pattern...");

    // Step 7a: POST to create job
    let inference_url = format!("http://127.0.0.1:{}/v1/inference", config.port);
    let payload = serde_json::json!({
        // NO job_id - server generates it
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.7
    });

    let create_response = match ureq::post(&inference_url)
        .set("Content-Type", "application/json")
        .send_json(&payload)
    {
        Ok(response) => response,
        Err(e) => {
            println!("❌ Job creation failed: {}", e);
            return Ok(());
        }
    };

    #[derive(serde::Deserialize)]
    struct CreateJobResponse {
        job_id: String,
        sse_url: String,
    }

    let job_response: CreateJobResponse = match create_response.into_json() {
        Ok(resp) => resp,
        Err(e) => {
            println!("❌ Failed to parse job response: {}", e);
            return Ok(());
        }
    };

    println!("✅ Job created: {}", job_response.job_id);
    println!("📡 SSE URL: {}", job_response.sse_url);

    // Step 7b: GET to stream results
    let stream_url = format!("http://127.0.0.1:{}{}", config.port, job_response.sse_url);

    match ureq::get(&stream_url).call() {
        Ok(response) => {
            println!("✅ SSE connection established");

            let reader = std::io::BufReader::new(response.into_reader());
            let mut token_count = 0;
            let mut done_received = false;

            println!("📡 Streaming tokens (30s timeout):");

            let stream_start = std::time::Instant::now();
            let stream_timeout = Duration::from_secs(30);

            for line in reader.lines() {
                // CRITICAL: Check timeout on EVERY line read!
                if stream_start.elapsed() > stream_timeout {
                    println!("\n\n❌ TIMEOUT: No tokens after 30 seconds - KILLING TEST!");
                    break;
                }

                if let Ok(line) = line {
                    if line.starts_with("data: ") {
                        let data = &line[6..];
                        if data == "[DONE]" {
                            done_received = true;
                            println!("\n✅ Received [DONE] signal");
                            break;
                        }
                        // Try to parse event
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                            // Check event type
                            if let Some(event_type) = json.get("type").and_then(|t| t.as_str()) {
                                if event_type == "token" {
                                    // Token event - extract text (field is "t" not "text")
                                    if let Some(text) = json.get("t").and_then(|t| t.as_str()) {
                                        print!("{}", text);
                                        std::io::Write::flush(&mut std::io::stdout()).ok();
                                        token_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            println!("\n\n📊 Inference Test Results");
            println!("==================================");
            println!("Tokens received: {}", token_count);
            println!("[DONE] signal: {}", if done_received { "✅" } else { "❌" });

            if token_count > 0 && done_received {
                println!("\n✅ DUAL-CALL PATTERN TEST PASSED!");
            } else {
                println!("\n❌ DUAL-CALL PATTERN TEST FAILED!");
            }
        }
        Err(e) => {
            println!("❌ SSE connection failed: {}", e);
        }
    }

    println!("\n🧹 Cleaning up...");

    // Force kill worker process
    let _ = worker.process.kill();
    let _ = worker.process.wait();

    drop(hive_server);
    println!("✓ Cleanup complete");

    println!("\n✅ Isolation test complete!");
    Ok(())
}
