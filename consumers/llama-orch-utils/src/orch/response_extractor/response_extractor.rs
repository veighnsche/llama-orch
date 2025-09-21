// Placeholder module for applet orch/response_extractor.
use crate::llm::invoke::InvokeResult;

pub fn run(result: &InvokeResult) -> String {
    // Rule 1: choices[0].text
    if let Some(first) = result.choices.first() {
        if !first.text.is_empty() {
            return first.text.clone();
        }
    }
    // Rule 2: first non-empty choice text anywhere in the list
    if let Some(found) = result.choices.iter().find(|c| !c.text.is_empty()) {
        return found.text.clone();
    }

    // Rule 3: best-effort fallback
    String::new()
}
