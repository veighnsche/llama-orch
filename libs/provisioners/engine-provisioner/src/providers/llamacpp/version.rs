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
