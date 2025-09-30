// Placeholder module for applet fs/file_writer.
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteIn {
    pub path: String,
    pub text: String,
    pub create_dirs: bool,
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteOut {
    pub path: String,
    pub bytes_written: usize,
}

pub fn run(input: WriteIn) -> io::Result<WriteOut> {
    let p = Path::new(&input.path);
    if input.create_dirs {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
    }
    fs::write(p, input.text.as_bytes())?;
    Ok(WriteOut { path: input.path, bytes_written: input.text.len() })
}
