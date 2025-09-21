use super::*;
use crate::prompt::message::Source;
use tempfile::tempdir;
use std::fs;

#[test]
fn composes_multiple_items_in_order() {
    let input = ThreadIn {
        items: vec![
            ThreadItem { role: "system".into(), source: Source::Text("s1".into()), dedent: false },
            ThreadItem { role: "user".into(), source: Source::Lines(vec!["a".into(), "b".into()]), dedent: false },
            ThreadItem { role: "assistant".into(), source: Source::Text("ok".into()), dedent: false },
        ],
    };
    let out = run(input).expect("run ok");
    assert_eq!(out.messages.len(), 3);
    assert_eq!(out.messages[0].role, "system");
    assert_eq!(out.messages[0].content, "s1");
    assert_eq!(out.messages[1].role, "user");
    assert_eq!(out.messages[1].content, "a\nb");
    assert_eq!(out.messages[2].role, "assistant");
    assert_eq!(out.messages[2].content, "ok");
}

#[test]
fn supports_mixed_sources() {
    let dir = tempdir().expect("tempdir");
    let file_path = dir.path().join("seed.txt");
    fs::write(&file_path, "héllo").expect("write");

    let input = ThreadIn {
        items: vec![
            ThreadItem { role: "system".into(), source: Source::File(file_path.to_string_lossy().into_owned()), dedent: false },
            ThreadItem { role: "user".into(), source: Source::Text("Q".into()), dedent: false },
            ThreadItem { role: "assistant".into(), source: Source::Lines(vec!["x".into(), "y".into()]), dedent: false },
        ],
    };
    let out = run(input).expect("run ok");
    assert_eq!(out.messages[0].content, "héllo");
    assert_eq!(out.messages[1].content, "Q");
    assert_eq!(out.messages[2].content, "x\ny");
}

#[test]
fn applies_dedent_per_item() {
    let input = ThreadIn {
        items: vec![
            ThreadItem { role: "system".into(), source: Source::Text("    one\n      two".into()), dedent: true },
            ThreadItem { role: "user".into(), source: Source::Lines(vec!["    a".into(), "    b".into()]), dedent: true },
        ],
    };
    let out = run(input).expect("run ok");
    assert_eq!(out.messages[0].content, "one\n  two");
    assert_eq!(out.messages[1].content, "a\nb");
}

#[test]
fn propagates_error_from_missing_file() {
    let input = ThreadIn {
        items: vec![
            ThreadItem { role: "system".into(), source: Source::File("/nonexistent/definitely.txt".into()), dedent: false },
        ],
    };
    let err = run(input).err().expect("expected error");
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

