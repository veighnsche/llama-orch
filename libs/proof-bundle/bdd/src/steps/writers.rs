use cucumber::{given, then, when};
use std::fs;
use std::io::{BufRead, BufReader, Read};

use crate::steps::world::BddWorld;

#[when(regex = r#"^I ensure dir \"([^\"]+)\"$"#)]
pub async fn when_ensure_dir(world: &mut BddWorld, sub: String) {
    let pb = world.get_pb().expect("bundle in world");
    pb.ensure_dir(&sub).expect("ensure dir");
}

#[then(regex = r#"^dir exists \"([^\"]+)\"$"#)]
pub async fn then_dir_exists(world: &mut BddWorld, sub: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(sub);
    assert!(p.is_dir(), "{} should be a directory", p.display());
}

#[when(regex = r#"^I overwrite markdown \"([^\"]+)\" with body \"([^\"]*)\"$"#)]
pub async fn when_overwrite_markdown(world: &mut BddWorld, name: String, body: String) {
    let pb = world.get_pb().expect("bundle in world");
    pb.write_markdown(name, &body).expect("write markdown");
}

#[when(regex = r#"^I write json file base \"([^\"]+)\" value (\{.*\})$"#)]
pub async fn when_write_json_base(world: &mut BddWorld, base: String, json_str: String) {
    let pb = world.get_pb().expect("bundle in world");
    let value: serde_json::Value = serde_json::from_str(&json_str).expect("parse json");
    pb.write_json(base, &value).expect("write json");
}

#[when(regex = r#"^I write json with meta base \"([^\"]+)\" value (\{.*\})$"#)]
pub async fn when_write_json_with_meta(world: &mut BddWorld, base: String, json_str: String) {
    let pb = world.get_pb().expect("bundle in world");
    let value: serde_json::Value = serde_json::from_str(&json_str).expect("parse json");
    pb.write_json_with_meta(base, &value).expect("write json with meta");
}

#[then(regex = r#"^json file \"([^\"]+)\" has field \"([^\"]+)\" equals \"([^\"]*)\"$"#)]
pub async fn then_json_field_equals(world: &mut BddWorld, name: String, field: String, expected: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let f = fs::File::open(&p).expect("open file");
    let v: serde_json::Value = serde_json::from_reader(f).expect("parse json file");
    let got = v.get(&field).expect("field present");
    let got_str = if got.is_string() { got.as_str().unwrap().to_string() } else { got.to_string() };
    assert_eq!(got_str, expected, "json field mismatch for {}", p.display());
}

#[then(regex = r#"^file \"([^\"]+)\" has at least (\d+) lines$"#)]
pub async fn file_has_min_lines(world: &mut BddWorld, name: String, min: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let f = fs::File::open(&p).expect("open file");
    let lines: Vec<_> = BufReader::new(f).lines().collect();
    let min_n: usize = min.parse().unwrap();
    assert!(lines.len() >= min_n, "{} should have at least {} lines", p.display(), min_n);
}

#[then(regex = r#"^file \"([^\"]+)\" has exactly (\d+) lines$"#)]
pub async fn file_has_exact_lines(world: &mut BddWorld, name: String, count: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let f = fs::File::open(&p).expect("open file");
    let lines: Vec<_> = BufReader::new(f).lines().collect();
    let n: usize = count.parse().unwrap();
    assert_eq!(lines.len(), n, "{} should have {} lines", p.display(), n);
}

#[then(regex = r#"^file \"([^\"]+)\" ends with a newline$"#)]
pub async fn file_ends_with_newline(world: &mut BddWorld, name: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let mut f = fs::File::open(&p).expect("open file");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).expect("read file");
    assert!(!buf.is_empty(), "file should not be empty");
    assert_eq!(*buf.last().unwrap(), b'\n', "{} should end with newline", p.display());
}

#[then(regex = r#"^second line of \"([^\"]+)\" equals \"([^\"]*)\"$"#)]
pub async fn second_line_equals(world: &mut BddWorld, name: String, expected: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let f = fs::File::open(&p).expect("open file");
    let mut it = BufReader::new(f).lines();
    let _first = it.next();
    let second = it.next().expect("has second line").expect("read line");
    assert_eq!(second, expected, "second line mismatch for {}", p.display());
}

#[then(regex = r#"^file \"([^\"]+)\" contains \"([^\"]+)\"$"#)]
pub async fn file_contains(world: &mut BddWorld, name: String, needle: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let mut f = fs::File::open(&p).expect("open file");
    let mut s = String::new();
    use std::io::Read as _;
    f.read_to_string(&mut s).expect("read file");
    assert!(s.contains(&needle), "file {} should contain {}", p.display(), needle);
}

#[then(regex = r#"^line (\d+) of \"([^\"]+)\" equals \"([^\"]*)\"$"#)]
pub async fn line_n_equals(world: &mut BddWorld, n: String, name: String, expected: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let f = fs::File::open(&p).expect("open file");
    let lines: Vec<_> = BufReader::new(f).lines().map(|l| l.expect("read line")).collect();
    let idx: usize = n.parse().unwrap();
    assert!(idx >= 1 && idx <= lines.len(), "line index out of bounds for {}", p.display());
    assert_eq!(lines[idx - 1], expected, "line {} mismatch for {}", idx, p.display());
}

#[then(regex = r#"^last line of \"([^\"]+)\" equals \"([^\"]*)\"$"#)]
pub async fn last_line_equals(world: &mut BddWorld, name: String, expected: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let f = fs::File::open(&p).expect("open file");
    let lines: Vec<_> = BufReader::new(f).lines().map(|l| l.expect("read line")).collect();
    assert!(!lines.is_empty(), "file should not be empty: {}", p.display());
    assert_eq!(lines.last().unwrap(), &expected, "last line mismatch for {}", p.display());
}
