use super::*;
use std::fs;
use tempfile::tempdir;

#[test]
fn builds_from_text_without_dedent() {
    let input = MessageIn { role: "user".into(), source: Source::Text("hello".into()), dedent: false };
    let out = run(input).expect("run ok");
    assert_eq!(out.role, "user");
    assert_eq!(out.content, "hello");
}

#[test]
fn builds_from_lines_joins_with_newlines() {
    let input = MessageIn { role: "user".into(), source: Source::Lines(vec!["a".into(), "b".into(), "c".into()]), dedent: false };
    let out = run(input).expect("run ok");
    assert_eq!(out.content, "a\nb\nc");
}

#[test]
fn builds_from_file_utf8_lossy() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("seed.txt");
    fs::write(&path, "héllo world").expect("write");
    let input = MessageIn { role: "system".into(), source: Source::File(path.to_string_lossy().into_owned()), dedent: false };
    let out = run(input).expect("run ok");
    assert_eq!(out.role, "system");
    assert_eq!(out.content, "héllo world");
}

#[test]
fn applies_dedent_when_true() {
    let text = "    line1\n      line2\n    line3";
    let input = MessageIn { role: "user".into(), source: Source::Text(text.into()), dedent: true };
    let out = run(input).expect("run ok");
    // Common leading spaces = 4 across non-empty lines, so remove 4 from each
    assert_eq!(out.content, "line1\n  line2\nline3");
}

#[test]
fn errors_on_missing_file() {
    let input = MessageIn { role: "user".into(), source: Source::File("/definitely/not/here.txt".into()), dedent: false };
    let err = run(input).err().expect("expected error");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}
