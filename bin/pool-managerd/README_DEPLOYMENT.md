# pool-managerd Deployment Guide

## Systemd Service (Home Profile)

### Installation

1. **Build the binary:**
   ```bash
   cargo build --release -p pool-managerd
   sudo cp target/release/pool-managerd /usr/local/bin/
   ```

2. **Create user and directories:**
   ```bash
   sudo useradd -r -s /bin/false llama-orch
   sudo mkdir -p /opt/llama-orch/.runtime
   sudo chown -R llama-orch:llama-orch /opt/llama-orch
   ```

3. **Install systemd unit:**
   ```bash
   sudo cp bin/pool-managerd/pool-managerd.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

4. **Start the service:**
   ```bash
   sudo systemctl enable pool-managerd
   sudo systemctl start pool-managerd
   ```

5. **Check status:**
   ```bash
   sudo systemctl status pool-managerd
   sudo journalctl -u pool-managerd -f
   ```

### Configuration

Environment variables (edit `/etc/systemd/system/pool-managerd.service`):

- `POOL_MANAGERD_ADDR` — Bind address (default: `127.0.0.1:9200`)
- `RUST_LOG` — Log level (default: `info`)

### API Endpoints

- `GET http://127.0.0.1:9200/health` — Health check
- `POST http://127.0.0.1:9200/pools/{id}/preload` — Spawn engine
- `GET http://127.0.0.1:9200/pools/{id}/status` — Pool status

### Testing

```bash
# Health check
curl http://127.0.0.1:9200/health

# Expected response:
# {"status":"ok","version":"0.0.0"}
```

## Docker (Alternative)

```dockerfile
FROM rust:1.75 as builder
WORKDIR /build
COPY . .
RUN cargo build --release -p pool-managerd

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/pool-managerd /usr/local/bin/
EXPOSE 9200
CMD ["pool-managerd"]
```

Build and run:
```bash
docker build -t pool-managerd -f bin/pool-managerd/Dockerfile .
docker run -p 9200:9200 -v /opt/llama-orch/.runtime:/runtime pool-managerd
```

## Kubernetes (Cloud Profile)

See `.specs/proposals/CLOUD_PROFILE.md` for DaemonSet deployment.

Example:
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: pool-managerd
spec:
  selector:
    matchLabels:
      app: pool-managerd
  template:
    metadata:
      labels:
        app: pool-managerd
    spec:
      containers:
      - name: pool-managerd
        image: pool-managerd:latest
        ports:
        - containerPort: 9200
        env:
        - name: POOL_MANAGERD_ADDR
          value: "0.0.0.0:9200"
        volumeMounts:
        - name: runtime
          mountPath: /runtime
      volumes:
      - name: runtime
        emptyDir: {}
```
