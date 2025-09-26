use cucumber::{given, then, when};
use std::fs;
use std::io::{BufRead, BufReader};

use crate::steps::world::BddWorld;
use proof_bundle::TestType;

fn parse_type(s: &str) -> TestType {
    match s.to_lowercase().as_str() {
        "unit" => TestType::Unit,
        "integration" => TestType::Integration,
        "contract" => TestType::Contract,
        "bdd" => TestType::Bdd,
        "determinism" => TestType::Determinism,
        "home-profile-smoke" | "smoke" => TestType::Smoke,
        "e2e-haiku" | "haiku" => TestType::E2eHaiku,
        other => panic!("unknown test type: {}", other),
    }
}

#[given(regex = r#"^a proof bundle of type \"([^\"]+)\" with run id \"([^\"]+)\"$"#)]
pub async fn given_bundle(world: &mut BddWorld, t: String, run_id: String) {
    world.set_env_for_bundle(&run_id);
    let tt = parse_type(&t);
    let pb = world.open_bundle(tt).expect("open bundle");
    assert!(pb.root().exists(), "bundle root should exist");
}

#[when(regex = r#"^I write markdown \"([^\"]+)\" body \"([^\"]*)\"$"#)]
pub async fn when_write_markdown(world: &mut BddWorld, name: String, body: String) {
    let pb = world.get_pb().expect("bundle in world");
    pb.write_markdown(name, &body).expect("write markdown");
}

#[when(regex = r#"^I write markdown with header \"([^\"]+)\" body \"([^\"]*)\"$"#)]
pub async fn when_write_markdown_with_header(world: &mut BddWorld, name: String, body: String) {
    let pb = world.get_pb().expect("bundle in world");
    pb.write_markdown_with_header(name, &body).expect("write markdown with header");
}

#[then(regex = r#"^file \"([^\"]+)\" first line equals \"([^\"]+)\"$"#)]
pub async fn then_first_line_equals(world: &mut BddWorld, name: String, expected: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let f = fs::File::open(&p).expect("open file");
    let mut it = BufReader::new(f).lines();
    let first = it.next().expect("has first line").expect("read line");
    assert_eq!(first, expected, "first line mismatch for {}", p.display());
}

#[when(regex = r#"^I write json \"([^\"]+)\" value (\{.*\})$"#)]
pub async fn when_write_json(world: &mut BddWorld, name: String, json_str: String) {
    let pb = world.get_pb().expect("bundle in world");
    let value: serde_json::Value = serde_json::from_str(&json_str).expect("parse json");
    pb.write_json(name, &value).expect("write json");
}

#[then(regex = r#"^file exists \"([^\"]+)\"$"#)]
pub async fn then_file_exists(world: &mut BddWorld, name: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    assert!(p.exists(), "{} should exist", p.display());
}

#[when(regex = r#"^I append ndjson \"([^\"]+)\" value (\{.*\})$"#)]
pub async fn when_append_ndjson(world: &mut BddWorld, name: String, json_str: String) {
    let pb = world.get_pb().expect("bundle in world");
    let value: serde_json::Value = serde_json::from_str(&json_str).expect("parse json");
    pb.append_ndjson(name, &value).expect("append ndjson");
}

#[then(regex = r#"^first line of \"([^\"]+)\" contains \"([^\"]+)\"$"#)]
pub async fn then_first_line_contains(world: &mut BddWorld, name: String, needle: String) {
    let pb = world.get_pb().expect("bundle in world");
    let p = pb.root().join(name);
    let f = fs::File::open(&p).expect("open file");
    let mut it = BufReader::new(f).lines();
    let first = it.next().expect("has first line").expect("read line");
    assert!(first.contains(&needle), "first line should contain {}", needle);
}

#[when(regex = r#"^I ensure ndjson meta \"([^\"]+)\"$"#)]
pub async fn when_ensure_ndjson_meta(world: &mut BddWorld, name: String) {
    let pb = world.get_pb().expect("bundle in world");
    pb.append_ndjson_autogen_meta(name).expect("append ndjson autogen meta");
}

#[when(regex = r#"^I write meta sibling for \"([^\"]+)\"$"#)]
pub async fn when_write_meta_sibling(world: &mut BddWorld, name: String) {
    let pb = world.get_pb().expect("bundle in world");
    pb.write_meta_sibling(name).expect("write meta sibling");
}
