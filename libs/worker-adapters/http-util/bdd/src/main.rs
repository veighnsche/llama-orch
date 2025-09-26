mod steps;

use cucumber::World as _;
use steps::world::BddWorld;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // Ensure proof bundles are emitted under the library crate root, not the bdd sub-crate
    if std::env::var("LLORCH_PROOF_DIR").is_err() {
        if let Some(parent) = root.parent() {
            let dir = parent.join(".proof_bundle");
            std::env::set_var("LLORCH_PROOF_DIR", dir);
        }
    }
    let features_env = std::env::var("LLORCH_BDD_FEATURE_PATH").ok();
    let features = if let Some(p) = features_env {
        let pb = std::path::PathBuf::from(p);
        if pb.is_absolute() {
            pb
        } else {
            root.join(pb)
        }
    } else {
        root.join("tests/features")
    };

    BddWorld::cucumber().fail_on_skipped().run_and_exit(features).await;
}
