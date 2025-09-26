use super::*;
use std::fs;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn reads_single_file_text_utf8() {
    let dir = tempdir().expect("tempdir");
    let file_path = dir.path().join("a.txt");
    fs::write(&file_path, "hello κόσμε").expect("write");

    let req = ReadRequest {
        paths: vec![file_path.to_string_lossy().into_owned()],
        as_text: true,
        encoding: None,
    };
    let resp = run(req).expect("run ok");
    assert_eq!(resp.files.len(), 1);
    let f = &resp.files[0];
    assert_eq!(f.path, file_path.to_string_lossy());
    assert!(f.bytes.is_none());
    assert_eq!(f.content.as_deref(), Some("hello κόσμε"));
}

#[test]
fn reads_multiple_files_text() {
    let dir = tempdir().expect("tempdir");
    let file1 = dir.path().join("one.txt");
    let file2 = dir.path().join("two.txt");
    fs::write(&file1, "first").expect("write1");
    fs::write(&file2, "second").expect("write2");

    let req = ReadRequest {
        paths: vec![file1.to_string_lossy().into_owned(), file2.to_string_lossy().into_owned()],
        as_text: true,
        encoding: Some("utf-8".into()),
    };
    let resp = run(req).expect("run ok");
    assert_eq!(resp.files.len(), 2);
    assert_eq!(resp.files[0].content.as_deref(), Some("first"));
    assert_eq!(resp.files[1].content.as_deref(), Some("second"));
}

#[test]
fn reads_binary_when_as_text_false() {
    let dir = tempdir().expect("tempdir");
    let file_path = dir.path().join("bin.dat");
    let bytes: Vec<u8> = vec![0xde, 0xad, 0xbe, 0xef, 0x00, 0x01];
    let mut f = fs::File::create(&file_path).expect("create");
    f.write_all(&bytes).expect("write");

    let req = ReadRequest {
        paths: vec![file_path.to_string_lossy().into_owned()],
        as_text: false,
        encoding: None,
    };
    let resp = run(req).expect("run ok");
    assert_eq!(resp.files.len(), 1);
    let b = resp.files[0].bytes.as_ref().expect("bytes");
    assert_eq!(b.len(), bytes.len());
    assert_eq!(&b[..4], &bytes[..4]);
    assert!(resp.files[0].content.is_none());
}

#[test]
fn errors_on_missing_file() {
    let dir = tempdir().expect("tempdir");
    let missing = dir.path().join("missing.txt");
    let req = ReadRequest {
        paths: vec![missing.to_string_lossy().into_owned()],
        as_text: true,
        encoding: None,
    };
    let err = run(req).err().expect("expected error");
    // io::ErrorKind should be NotFound
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}
