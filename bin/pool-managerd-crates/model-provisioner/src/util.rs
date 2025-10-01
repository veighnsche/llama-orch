use anyhow::Result;
use std::fs;
use std::path::Path;

pub(crate) fn compute_sha256_hex(path: &Path) -> Result<String> {
    use sha2::{Digest as ShaDigest, Sha256};
    use std::io::Read;
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let out = hasher.finalize();
    Ok(hex::encode(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_small_file_is_stable() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("x.bin");
        fs::write(&p, b"abc").unwrap();
        let h = compute_sha256_hex(&p).unwrap();
        // precomputed sha256 of "abc"
        assert_eq!(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
