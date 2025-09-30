use std::net::TcpStream;

/// Best-effort: GET /version and parse JSON body for version-like fields.
pub fn try_fetch_engine_version(host: &str, port: u16) -> Option<String> {
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(addr).ok()?;
    let req = format!(
        "GET /version HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        host
    );
    let _ = std::io::Write::write_all(&mut stream, req.as_bytes());
    let mut buf = Vec::new();
    let _ = std::io::Read::read_to_end(&mut stream, &mut buf);
    let text = String::from_utf8_lossy(&buf);
    if let Some(idx) = text.find("\r\n\r\n") {
        let body = &text[idx + 4..];
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
            if let Some(s) = v.get("version").and_then(|x| x.as_str()) {
                return Some(s.to_string());
            }
            if let Some(s) = v.get("git_describe").and_then(|x| x.as_str()) {
                return Some(s.to_string());
            }
            if let Some(s) = v.get("build").and_then(|x| x.as_str()) {
                return Some(s.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::try_fetch_engine_version;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    fn serve_once(body: &str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let body = body.to_string();
        thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                let _ = s.write_all(resp.as_bytes());
            }
        });
        port
    }

    #[test]
    fn parses_version_field() {
        let port = serve_once("{\"version\":\"v1.2.3\"}");
        let got = try_fetch_engine_version("127.0.0.1", port);
        assert_eq!(got.as_deref(), Some("v1.2.3"));
    }

    #[test]
    fn falls_back_to_git_describe_then_build() {
        let port1 = serve_once("{\"git_describe\":\"g123\"}");
        assert_eq!(try_fetch_engine_version("127.0.0.1", port1).as_deref(), Some("g123"));
        let port2 = serve_once("{\"build\":\"abc\"}");
        assert_eq!(try_fetch_engine_version("127.0.0.1", port2).as_deref(), Some("abc"));
    }

    #[test]
    fn returns_none_on_invalid_json() {
        let port = serve_once("not-json");
        let got = try_fetch_engine_version("127.0.0.1", port);
        assert!(got.is_none());
    }
}

