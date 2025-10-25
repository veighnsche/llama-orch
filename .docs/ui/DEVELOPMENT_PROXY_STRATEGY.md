# Development Proxy Strategy

**TEAM-293: Backend proxies to frontend dev servers during development**

## Overview

During development, backend binaries (queen, hive, workers) should **proxy** requests to Vite dev servers instead of serving static files. This enables hot-reload and faster development.

## Architecture

### Production (Static Files)
```
http://localhost:7833/      → queen-rbee API
http://localhost:7833/ui    → Static files (dist/)
```

### Development (Proxy to Vite)
```
http://localhost:7833/      → queen-rbee API
http://localhost:7833/ui    → Proxy to http://localhost:7834 (Vite dev server)
```

## Port Mapping

| Binary | API Port | UI Dev Server | Proxy Target |
|--------|----------|---------------|--------------|
| queen-rbee | 7833 | 7834 | `http://localhost:7834` |
| rbee-hive | 7835 | 7836 | `http://localhost:7836` |
| llm-worker | 8080 | 7837 | `http://localhost:7837` |
| comfy-worker | 8188 | 7838 | `http://localhost:7838` |
| vllm-worker | 8000 | 7839 | `http://localhost:7839` |

## Implementation

### Rust Backend (Axum)

**File:** `bin/10_queen_rbee/src/main.rs`

```rust
use axum::{Router, routing::get};
use tower_http::services::ServeDir;

#[derive(clap::Parser)]
struct Args {
    #[arg(short, long, default_value = "7833")]
    port: u16,
    
    #[arg(long, default_value = "false")]
    dev_mode: bool,
}

async fn main() -> Result<()> {
    let args = Args::parse();
    
    let app = Router::new()
        .route("/api/*", /* API routes */)
        .nest_service("/ui", ui_service(args.dev_mode));
    
    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}

fn ui_service(dev_mode: bool) -> Router {
    if dev_mode {
        // Development: Proxy to Vite dev server
        use axum::http::{Uri, Request};
        use tower::ServiceBuilder;
        
        Router::new().fallback(|req: Request<Body>| async move {
            let vite_url = "http://localhost:7834"; // Queen UI dev server
            let uri = format!("{}{}", vite_url, req.uri().path());
            
            // Proxy request to Vite
            let client = reqwest::Client::new();
            let response = client
                .request(req.method().clone(), &uri)
                .headers(req.headers().clone())
                .body(req.into_body())
                .send()
                .await?;
            
            Ok::<_, Infallible>(response)
        })
    } else {
        // Production: Serve static files
        Router::new().nest_service(
            "/",
            ServeDir::new("../../../frontend/apps/10_queen_rbee/dist")
        )
    }
}
```

### Alternative: Use Environment Variable

```rust
fn ui_service() -> Router {
    let dev_mode = std::env::var("DEV_MODE").is_ok();
    
    if dev_mode {
        // Proxy to Vite
        proxy_to_vite("http://localhost:7834")
    } else {
        // Serve static files
        ServeDir::new("../../../frontend/apps/10_queen_rbee/dist")
    }
}
```

### Using tower-http Proxy (Recommended)

**Add dependency:**
```toml
[dependencies]
tower-http = { version = "0.5", features = ["fs", "trace"] }
hyper = { version = "1.0", features = ["client", "http1"] }
```

**Implementation:**
```rust
use tower_http::services::ServeDir;
use hyper::client::HttpConnector;

fn ui_service(dev_mode: bool, vite_url: &str) -> Router {
    if dev_mode {
        // Proxy to Vite dev server
        let client = hyper::Client::new();
        
        Router::new().fallback(move |req: Request<Body>| {
            let client = client.clone();
            let vite_url = vite_url.to_string();
            
            async move {
                let uri = format!("{}{}", vite_url, req.uri().path());
                let mut proxy_req = Request::builder()
                    .method(req.method())
                    .uri(uri)
                    .body(req.into_body())
                    .unwrap();
                
                *proxy_req.headers_mut() = req.headers().clone();
                
                client.request(proxy_req).await
            }
        })
    } else {
        // Serve static files
        Router::new().nest_service(
            "/",
            ServeDir::new("../../../frontend/apps/10_queen_rbee/dist")
        )
    }
}
```

## Development Workflow

### Terminal 1: Start Vite Dev Server
```bash
cd frontend/apps/10_queen_rbee
pnpm dev  # Runs on port 7834
```

### Terminal 2: Start Backend in Dev Mode
```bash
cd bin/10_queen_rbee
cargo run -- --dev-mode
# or
DEV_MODE=1 cargo run
```

### Access
```bash
# Both work, backend proxies to Vite:
curl http://localhost:7833/ui  # Backend (proxies to 7834)
curl http://localhost:7834      # Vite dev server (direct)
```

## Benefits

✅ **Hot Reload:** Changes in React code instantly reflect  
✅ **Fast Refresh:** Vite's HMR works seamlessly  
✅ **Single URL:** Access everything through backend URL  
✅ **No CORS:** Backend and frontend on same origin  
✅ **Production-like:** Same URL structure as production

## Configuration Per Binary

### queen-rbee
```rust
// bin/10_queen_rbee/src/main.rs
const VITE_DEV_SERVER: &str = "http://localhost:7834";
const UI_DIST_PATH: &str = "../../../frontend/apps/10_queen_rbee/dist";
```

### rbee-hive
```rust
// bin/20_rbee_hive/src/main.rs
const VITE_DEV_SERVER: &str = "http://localhost:7836";
const UI_DIST_PATH: &str = "../../../frontend/apps/20_rbee_hive/dist";
```

### llm-worker
```rust
// bin/30_llm_worker_rbee/src/main.rs
const VITE_DEV_SERVER: &str = "http://localhost:7837";
const UI_DIST_PATH: &str = "../../../frontend/apps/30_llm_worker_rbee/dist";
```

## Environment Variables

```bash
# Development
export DEV_MODE=1
export QUEEN_UI_DEV_SERVER=http://localhost:7834
export HIVE_UI_DEV_SERVER=http://localhost:7836
export WORKER_UI_DEV_SERVER=http://localhost:7837

# Production (default)
unset DEV_MODE
```

## Makefile Helper

**File:** `Makefile`

```makefile
.PHONY: dev-queen dev-hive dev-worker

dev-queen:
	@echo "Starting queen-rbee in dev mode..."
	@cd frontend/apps/10_queen_rbee && pnpm dev &
	@sleep 2
	@cd bin/10_queen_rbee && DEV_MODE=1 cargo run

dev-hive:
	@echo "Starting rbee-hive in dev mode..."
	@cd frontend/apps/20_rbee_hive && pnpm dev &
	@sleep 2
	@cd bin/20_rbee_hive && DEV_MODE=1 cargo run

dev-worker:
	@echo "Starting llm-worker in dev mode..."
	@cd frontend/apps/30_llm_worker_rbee && pnpm dev &
	@sleep 2
	@cd bin/30_llm_worker_rbee && DEV_MODE=1 cargo run
```

## Testing

### Development Mode
```bash
# Start Vite dev server
cd frontend/apps/10_queen_rbee
pnpm dev

# Start backend in dev mode
cd bin/10_queen_rbee
DEV_MODE=1 cargo run

# Test proxy
curl http://localhost:7833/ui  # Should return Vite dev server content
```

### Production Mode
```bash
# Build frontend
cd frontend/apps/10_queen_rbee
pnpm build

# Start backend (no dev mode)
cd bin/10_queen_rbee
cargo run

# Test static files
curl http://localhost:7833/ui  # Should return dist/ content
```

## Troubleshooting

### Issue: CORS Errors
**Solution:** Backend proxy handles CORS, no issues

### Issue: Hot Reload Not Working
**Solution:** Ensure Vite dev server is running first

### Issue: 404 on /ui
**Solution:** Check `DEV_MODE` is set and Vite server is running

### Issue: Slow Response
**Solution:** Vite dev server might be compiling, wait a moment

## Summary

**Development:**
- Backend proxies `/ui` to Vite dev server
- Hot reload works seamlessly
- Fast development cycle

**Production:**
- Backend serves static files from `dist/`
- No Vite dependency
- Optimized bundle

**Key Principle:** Same URL structure in both modes, different backend behavior.

---

**Status:** 📋 STRATEGY DEFINED  
**Next:** Implement proxy logic in each binary's main.rs
