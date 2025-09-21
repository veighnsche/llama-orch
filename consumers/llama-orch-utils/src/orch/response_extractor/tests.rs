use super::*;
use crate::llm::invoke::{Choice, InvokeResult};

#[test]
fn returns_first_choice_text_when_present() {
    let res = InvokeResult { choices: vec![Choice { text: "hello".into() }], usage: None };
    assert_eq!(run(&res), "hello");
}

#[test]
fn returns_first_nonempty_choice_text_when_first_is_empty() {
    let res = InvokeResult { choices: vec![Choice { text: "".into() }, Choice { text: "hi".into() }], usage: None };
    assert_eq!(run(&res), "hi");
}

#[test]
fn returns_empty_string_when_no_choices_or_text() {
    let res1 = InvokeResult { choices: vec![], usage: None };
    assert_eq!(run(&res1), "");

    let res2 = InvokeResult { choices: vec![Choice { text: "".into() }], usage: None };
    assert_eq!(run(&res2), "");
}

#[test]
fn preserves_whitespace_in_text() {
    let res = InvokeResult { choices: vec![Choice { text: "  spaced  ".into() }], usage: None };
    assert_eq!(run(&res), "  spaced  ");
}
