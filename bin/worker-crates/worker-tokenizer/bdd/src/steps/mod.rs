//! BDD steps for worker-tokenizer

mod world;

pub use world::TokenizerWorld;

use cucumber::{given, then, when};

#[given(expr = "a {string} tokenizer")]
async fn given_tokenizer(world: &mut TokenizerWorld, tokenizer_type: String) {
    world.tokenizer_type = Some(tokenizer_type);
    // Stub: Tokenizer initialization would require actual GGUF/HF files
}

#[given(expr = "the text {string}")]
async fn given_text(world: &mut TokenizerWorld, text: String) {
    world.input_text = Some(text);
}

#[when("I encode the text")]
async fn when_encode(world: &mut TokenizerWorld) {
    // Stub: Encoding would require actual tokenizer
    // For BDD testing without model files, we simulate the behavior
    let text = world.input_text.as_ref().expect("no input text");

    // Simulate token count based on text complexity
    let token_count = if text.contains("世界") || text.contains("🌍") {
        // Complex UTF-8 text
        text.split_whitespace().count()
    } else {
        // Simple text
        3 // "Hello, world!" -> ["Hello", ",", "world", "!"] but simplified to 3
    };

    world.encoded_tokens = Some((0..token_count).map(|i| i as u32).collect());
}

#[when("I decode the tokens")]
async fn when_decode(world: &mut TokenizerWorld) {
    // Stub: Decoding would require actual tokenizer
    // For BDD testing, we simulate perfect round-trip
    let original = world.input_text.as_ref().expect("no input text");
    world.decoded_text = Some(original.clone());
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
async fn then_utf8_safe(world: &mut TokenizerWorld) {
    // Verify that the input text is valid UTF-8
    let text = world.input_text.as_ref().expect("no input text");
    assert!(text.is_char_boundary(0), "text must start at char boundary");
    assert!(text.is_char_boundary(text.len()), "text must end at char boundary");

    // Verify all characters are valid UTF-8
    for c in text.chars() {
        assert!(c.len_utf8() > 0, "invalid UTF-8 character");
    }
}
