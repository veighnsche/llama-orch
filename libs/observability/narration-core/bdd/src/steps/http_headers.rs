// Step definitions for HTTP header propagation behaviors

use cucumber::{then, when};
use observability_narration_core::http::{extract_context_from_headers, inject_context_into_headers};
use crate::steps::world::World;

#[when(regex = r#"^I set header "([^"]+)" to "([^"]+)"$"#)]
pub async fn when_set_header(world: &mut World, name: String, value: String) {
    world.headers.insert(name, value);
}

#[when("I extract context from headers")]
pub async fn when_extract_context(world: &mut World) {
    let (correlation_id, trace_id, span_id, parent_span_id) = 
        extract_context_from_headers(&world.headers);
    world.extracted_correlation_id = correlation_id;
    world.extracted_trace_id = trace_id;
    world.extracted_span_id = span_id;
    world.extracted_parent_span_id = parent_span_id;
}

#[when(regex = r#"^I inject correlation_id "([^"]+)" into headers$"#)]
pub async fn when_inject_correlation_id(world: &mut World, correlation_id: String) {
    inject_context_into_headers(&mut world.headers, Some(&correlation_id), None, None, None);
}

#[when(regex = r#"^I inject trace_id "([^"]+)" into headers$"#)]
pub async fn when_inject_trace_id(world: &mut World, trace_id: String) {
    inject_context_into_headers(&mut world.headers, None, Some(&trace_id), None, None);
}

#[when(regex = r#"^I inject span_id "([^"]+)" into headers$"#)]
pub async fn when_inject_span_id(world: &mut World, span_id: String) {
    inject_context_into_headers(&mut world.headers, None, None, Some(&span_id), None);
}

#[when(regex = r#"^I inject parent_span_id "([^"]+)" into headers$"#)]
pub async fn when_inject_parent_span_id(world: &mut World, parent_span_id: String) {
    inject_context_into_headers(&mut world.headers, None, None, None, Some(&parent_span_id));
}

#[when("I inject correlation_id None into headers")]
pub async fn when_inject_none_correlation_id(world: &mut World) {
    inject_context_into_headers(&mut world.headers, None, None, None, None);
}

#[when("I inject all None fields into headers")]
pub async fn when_inject_all_none(world: &mut World) {
    world.headers.clear();
    inject_context_into_headers(&mut world.headers, None, None, None, None);
}

#[then(regex = r#"^extracted correlation_id is "([^"]+)"$"#)]
pub async fn then_extracted_correlation_id(world: &mut World, expected: String) {
    assert_eq!(world.extracted_correlation_id.as_ref(), Some(&expected));
}

#[then(regex = r#"^extracted trace_id is "([^"]+)"$"#)]
pub async fn then_extracted_trace_id(world: &mut World, expected: String) {
    assert_eq!(world.extracted_trace_id.as_ref(), Some(&expected));
}

#[then(regex = r#"^extracted span_id is "([^"]+)"$"#)]
pub async fn then_extracted_span_id(world: &mut World, expected: String) {
    assert_eq!(world.extracted_span_id.as_ref(), Some(&expected));
}

#[then(regex = r#"^extracted parent_span_id is "([^"]+)"$"#)]
pub async fn then_extracted_parent_span_id(world: &mut World, expected: String) {
    assert_eq!(world.extracted_parent_span_id.as_ref(), Some(&expected));
}

#[then("extracted correlation_id is None")]
pub async fn then_extracted_correlation_id_none(world: &mut World) {
    assert_eq!(world.extracted_correlation_id, None);
}

#[then("all extracted fields are None")]
pub async fn then_all_extracted_none(world: &mut World) {
    assert_eq!(world.extracted_correlation_id, None);
    assert_eq!(world.extracted_trace_id, None);
    assert_eq!(world.extracted_span_id, None);
    assert_eq!(world.extracted_parent_span_id, None);
}

#[then(regex = r#"^header "([^"]+)" is "([^"]+)"$"#)]
pub async fn then_header_is(world: &mut World, name: String, expected: String) {
    assert_eq!(world.headers.get(&name), Some(&expected));
}

#[then(regex = r#"^header "([^"]+)" is not present$"#)]
pub async fn then_header_not_present(world: &mut World, name: String) {
    assert!(!world.headers.contains_key(&name));
}

#[then("no headers are present")]
pub async fn then_no_headers(world: &mut World) {
    assert!(world.headers.is_empty());
}
