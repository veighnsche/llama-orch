use crate::steps::world::World;
use cucumber::{given, then};
use http::Method;

#[given(regex = r"^no API key is provided$")]
pub async fn given_no_api_key(world: &mut World) {
    world.push_fact("auth.none");
    world.api_key = None;
}

#[then(regex = r"^I receive 401 Unauthorized$")]
pub async fn then_401_unauthorized(world: &mut World) {
    // Drive a simple endpoint
    let _ = world.http_call(Method::GET, "/v1/capabilities", None).await;
    assert_eq!(world.last_status, Some(http::StatusCode::UNAUTHORIZED));
}

#[given(regex = r"^an invalid API key is provided$")]
pub async fn given_invalid_api_key(world: &mut World) {
    world.push_fact("auth.invalid");
    world.api_key = Some("invalid".into());
}

#[then(regex = r"^I receive 403 Forbidden$")]
pub async fn then_403_forbidden(world: &mut World) {
    let _ = world.http_call(Method::GET, "/v1/capabilities", None).await;
    assert_eq!(world.last_status, Some(http::StatusCode::FORBIDDEN));
    // Reset for subsequent steps
    world.api_key = Some("valid".into());
}
