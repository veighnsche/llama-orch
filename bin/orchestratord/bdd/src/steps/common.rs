// Common step definitions used across features
// Behaviors: Status code checks, common assertions

use crate::steps::world::World;
use cucumber::then;
use http::StatusCode;

// B-MW-004, B-DP-024, B-CAT-006, B-ART-004: Status code assertions
#[then(regex = "^I receive (\\d+)$")]
pub async fn then_status_code(world: &mut World, code: u16) {
    let expected = StatusCode::from_u16(code).expect("invalid status code");
    assert_eq!(world.last_status, Some(expected), "status code mismatch");
}

// Removed - conflicts with specific steps in control_plane.rs
// Use specific steps like "I receive 200 OK" instead

#[then(regex = "^I receive 200 OK$")]
pub async fn then_200_ok(world: &mut World) {
    then_status_code(world, 200).await;
}

#[then(regex = "^I receive 201 Created$")]
pub async fn then_201_created(world: &mut World) {
    then_status_code(world, 201).await;
}

#[then(regex = "^I receive 202 Accepted$")]
pub async fn then_202_accepted(world: &mut World) {
    then_status_code(world, 202).await;
}

#[then(regex = "^I receive 204 No Content$")]
pub async fn then_204_no_content(world: &mut World) {
    then_status_code(world, 204).await;
}

#[then(regex = "^I receive 400 Bad Request$")]
pub async fn then_400_bad_request(world: &mut World) {
    then_status_code(world, 400).await;
}

#[then(regex = "^I receive 404 Not Found$")]
pub async fn then_404_not_found(world: &mut World) {
    then_status_code(world, 404).await;
}
