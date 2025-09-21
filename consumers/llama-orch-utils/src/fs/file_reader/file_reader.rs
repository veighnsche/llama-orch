// Placeholder module for applet fs/file_reader.
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ReadRequest {
    pub paths: Vec<String>,
    pub as_text: bool,
    pub encoding: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FileBlob {
    pub path: String,
    pub content: Option<String>,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct ReadResponse {
    pub files: Vec<FileBlob>,
}

pub fn run(req: ReadRequest) -> io::Result<ReadResponse> {
    let mut out = Vec::with_capacity(req.paths.len());
    for p in req.paths.iter() {
        let path = Path::new(p);
        let data = fs::read(path)?;
        if req.as_text {
            // For M2, ignore encoding other than best-effort UTF-8.
            let s = String::from_utf8_lossy(&data).to_string();
            out.push(FileBlob {
                path: p.clone(),
                content: Some(s),
                bytes: None,
            });
        } else {
            out.push(FileBlob {
                path: p.clone(),
                content: None,
                bytes: Some(data),
            });
        }
    }
    Ok(ReadResponse { files: out })
}
