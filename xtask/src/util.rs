use anyhow::{anyhow, Result};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

pub fn repo_root() -> Result<PathBuf> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .ok_or_else(|| anyhow!("xtask: failed to locate repo root"))?
        .to_path_buf())
}

pub fn write_if_changed(path: &PathBuf, contents: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let write_needed = match fs::read_to_string(path) {
        Ok(existing) => existing != contents,
        Err(_) => true,
    };
    if write_needed {
        atomic_write(path, contents.as_bytes())?;
        println!("wrote {}", path.display());
    } else {
        println!("unchanged {}", path.display());
    }
    Ok(())
}

pub fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    let dir = path.parent().ok_or_else(|| anyhow!("no parent directory for {}", path.display()))?;
    fs::create_dir_all(dir)?;
    let tmp_path =
        dir.join(format!(".{}.tmp", path.file_name().and_then(|s| s.to_str()).unwrap_or("file")));

    // Write to temp file
    {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(data)?;
        f.sync_all()?;
    }

    // Try atomic rename first
    match fs::rename(&tmp_path, path) {
        Ok(_) => {}
        Err(e) => {
            // If cross-device link (EXDEV), fall back to copy
            let is_exdev = matches!(e.raw_os_error(), Some(code) if code == 18);
            if is_exdev {
                // Copy then remove tmp
                let mut dest = fs::File::create(path)?;
                let mut src = fs::File::open(&tmp_path)?;
                io::copy(&mut src, &mut dest)?;
                dest.sync_all()?;
                fs::remove_file(&tmp_path)?;
            } else {
                // Clean up tmp and propagate error
                let _ = fs::remove_file(&tmp_path);
                return Err(e.into());
            }
        }
    }

    // fsync the directory to persist filename changes
    let dir_file = fs::File::open(dir)?;
    dir_file.sync_all()?;
    Ok(())
}
