use cucumber::{given, then, when};

use crate::steps::world::BddWorld;
use proof_bundle::LegacyTestType;

fn parse_type(s: &str) -> LegacyTestType {
    match s.to_lowercase().as_str() {
        "unit" => LegacyTestType::Unit,
        "integration" => LegacyTestType::Integration,
        "contract" => LegacyTestType::Contract,
        "bdd" => LegacyTestType::Bdd,
        "determinism" => LegacyTestType::Determinism,
        "home-profile-smoke" | "smoke" => LegacyTestType::Smoke,
        "e2e-haiku" | "haiku" => LegacyTestType::E2eHaiku,
        other => panic!("unknown test type: {}", other),
    }
}

#[given(regex = r#"^I clear proof bundle env overrides$"#)]
pub async fn clear_env_overrides(_world: &mut BddWorld) {
    std::env::remove_var("LLORCH_PROOF_DIR");
    std::env::remove_var("LLORCH_RUN_ID");
}

#[given(regex = r#"^I set current dir to the world root$"#)]
pub async fn set_cwd_world_root(world: &mut BddWorld) {
    let root = world.ensure_root().to_path_buf();
    std::env::set_current_dir(&root).expect("set current dir");
}

#[when(regex = r#"^I open a bundle for type \"([^\"]+)\"$"#)]
pub async fn open_bundle(world: &mut BddWorld, t: String) {
    let tt = parse_type(&t);
    let pb = world.open_bundle(tt).expect("open bundle");
    assert!(pb.root().exists());
}

#[then(regex = r#"^bundle root ends with type dir \"([^\"]+)\" and a run id$"#)]
pub async fn bundle_root_has_type_and_run_id(world: &mut BddWorld, type_dir: String) {
    let pb = world.get_pb().expect("bundle in world");
    let root = pb.root();
    let run_id = root.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let parent = root.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
    assert_eq!(parent, type_dir, "type dir mismatch");
    assert!(!run_id.is_empty(), "run_id should be non-empty");
}

#[then(regex = r#"^bundle root is under world \.proof_bundle$"#)]
pub async fn bundle_root_under_world_proof_dir(world: &mut BddWorld) {
    let pb = world.get_pb().expect("bundle in world");
    let root = pb.root();
    let expected_base = world.ensure_root().join(".proof_bundle");
    assert!(
        root.starts_with(&expected_base),
        "{} should start with {}",
        root.display(),
        expected_base.display()
    );
}

#[given(regex = r#"^I set env overrides with run id \"([^\"]+)\"$"#)]
pub async fn set_env_overrides(world: &mut BddWorld, rid: String) {
    world.set_env_for_bundle(&rid);
}

#[then(regex = r#"^bundle root is under world root and ends with \"([^\"]+)\"/\"([^\"]+)\"$"#)]
pub async fn bundle_root_under_world(world: &mut BddWorld, type_dir: String, rid: String) {
    let pb = world.get_pb().expect("bundle in world");
    let root = pb.root();
    let w = world.ensure_root();
    assert!(root.starts_with(w), "root should be under world root");
    let parent = root.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
    let run_id = root.file_name().and_then(|s| s.to_str()).unwrap_or("");
    assert_eq!(parent, type_dir, "type dir mismatch");
    assert_eq!(run_id, rid, "run_id mismatch");
}

#[then(regex = r#"^dir name mapping for type \"([^\"]+)\" equals \"([^\"]+)\"$"#)]
pub async fn mapping_for_type(_world: &mut BddWorld, type_name: String, expected: String) {
    let tt = parse_type(&type_name);
    assert_eq!(tt.as_dir(), expected.as_str());
}

#[then(regex = r#"^the run id matches regex \"([^\"]+)\"$"#)]
pub async fn run_id_matches(world: &mut BddWorld, pat: String) {
    let pb = world.get_pb().expect("bundle in world");
    let root = pb.root();
    let run_id = root.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let re = regex::Regex::new(&pat).expect("compile regex");
    assert!(re.is_match(run_id), "run_id {} does not match {}", run_id, pat);
}
