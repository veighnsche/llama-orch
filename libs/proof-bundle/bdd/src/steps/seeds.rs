use cucumber::{then, when};
use std::fs;
use std::io::{BufRead, BufReader, Read};

use crate::steps::world::BddWorld;

#[when(regex = r#"^I record seed \"([^\"]+)\"$"#)]
pub async fn when_record_seed(world: &mut BddWorld, seed: String) {
    let pb = world.get_pb().expect("bundle in world");
    pb.seeds().record(seed).expect("record seed");
}

#[then(regex = r#"^seeds file exists$"#)]
pub async fn seeds_file_exists(world: &mut BddWorld) {
    let pb = world.get_pb().expect("bundle in world");
    assert!(pb.root().join("seeds.txt").exists());
}
