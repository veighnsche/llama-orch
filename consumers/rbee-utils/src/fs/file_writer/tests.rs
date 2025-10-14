use super::*;
use std::fs;
use std::io::Read;
use tempfile::tempdir;

#[test]
fn writes_new_file() {
    let dir = tempdir().expect("tempdir");
    let target = dir.path().join("new.txt");

    let input = WriteIn {
        path: target.to_string_lossy().into_owned(),
        text: "hello".into(),
        create_dirs: false,
    };
    let out = run(input).expect("run ok");
    assert_eq!(out.path, target.to_string_lossy());
    assert_eq!(out.bytes_written, "hello".as_bytes().len());

    let mut s = String::new();
    fs::File::open(&target).expect("open").read_to_string(&mut s).expect("read");
    assert_eq!(s, "hello");
}

#[test]
fn overwrites_existing_file() {
    let dir = tempdir().expect("tempdir");
    let target = dir.path().join("file.txt");
    fs::write(&target, "AAA").expect("seed");

    let text = "BBBCC";
    let input = WriteIn {
        path: target.to_string_lossy().into_owned(),
        text: text.into(),
        create_dirs: false,
    };
    let out = run(input).expect("run ok");
    assert_eq!(out.bytes_written, text.as_bytes().len());

    let on_disk = fs::read_to_string(&target).expect("read back");
    assert_eq!(on_disk, text);
}

#[test]
fn creates_parent_dirs_when_requested() {
    let dir = tempdir().expect("tempdir");
    let target = dir.path().join("a/b/c.txt");
    let input = WriteIn {
        path: target.to_string_lossy().into_owned(),
        text: "data".into(),
        create_dirs: true,
    };
    let out = run(input).expect("run ok");
    assert_eq!(out.bytes_written, 4);
    assert!(target.exists());
    let on_disk = fs::read_to_string(&target).expect("read back");
    assert_eq!(on_disk, "data");
}

#[test]
fn errors_on_invalid_path_when_dirs_missing() {
    let dir = tempdir().expect("tempdir");
    let target = dir.path().join("no/parents/here.txt");
    let input = WriteIn {
        path: target.to_string_lossy().into_owned(),
        text: "x".into(),
        create_dirs: false,
    };
    let err = run(input).err().expect("expected error");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}
