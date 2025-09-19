mod steps;

use cucumber::World as _;
use steps::world::World;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let features_env = std::env::var("LLORCH_BDD_FEATURE_PATH").ok();
    let features = if let Some(p) = features_env {
        let pb = std::path::PathBuf::from(p);
        if pb.is_absolute() { pb } else { root.join(pb) }
    } else {
        root.join("tests/features")
    };

    World::cucumber()
        .run_and_exit(features)
        .await;
}
