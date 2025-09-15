use crate::steps::world::World;
use cucumber::{given, when};

#[when(regex = r"^when apply runs$")]
pub async fn when_apply(world: &mut World) {
    world.push_fact("apply");
}

#[given(regex = r"^target filesystem is unsupported$")]
pub async fn given_target_fs_unsupported(world: &mut World) {
    world.push_fact("apply.target_fs_unsupported");
}
