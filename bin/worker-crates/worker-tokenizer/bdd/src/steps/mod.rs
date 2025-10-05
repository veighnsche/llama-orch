//! BDD steps for worker-tokenizer

mod world;

pub use world::TokenizerWorld;

use cucumber::{given, then, when};

#[given(expr = "a {string} tokenizer")]
async fn given_tokenizer(world: &mut TokenizerWorld, tokenizer_type: String) {
    world.tokenizer_type = Some(tokenizer_type);
    // TODO: Initialize tokenizer based on type
}

#[given(expr = "the text {string}")]
async fn given_text(world: &mut TokenizerWorld, text: String) {
    world.input_text = Some(text);
}

#[when("I encode the text")]
async fn when_encode(world: &mut TokenizerWorld) {
    // TODO: Implement encoding
    world.encoded_tokens = Some(vec![1, 2, 3]); // Placeholder
}

#[when("I decode the tokens")]
async fn when_decode(world: &mut TokenizerWorld) {
    // TODO: Implement decoding
    world.decoded_text = Some("decoded".to_string()); // Placeholder
}

#[then(expr = "the token count should be {int}")]
async fn then_token_count(world: &mut TokenizerWorld, expected: usize) {
    let tokens = world.encoded_tokens.as_ref().expect("no tokens");
    assert_eq!(tokens.len(), expected, "token count mismatch");
}

#[then("the decoded text should match the original")]
async fn then_decoded_matches(world: &mut TokenizerWorld) {
    let original = world.input_text.as_ref().expect("no input text");
    let decoded = world.decoded_text.as_ref().expect("no decoded text");
    assert_eq!(decoded, original, "decoded text mismatch");
}

#[then("the encoding should be UTF-8 safe")]
async fn then_utf8_safe(_world: &mut TokenizerWorld) {
    // TODO: Verify UTF-8 safety
}
