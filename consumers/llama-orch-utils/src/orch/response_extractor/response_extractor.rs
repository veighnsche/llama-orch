// Placeholder module for applet orch/response_extractor.
use crate::llm::invoke::InvokeResult;

pub fn run(result: &InvokeResult) -> String {
    // Rule 1: choices[0].text
    if let Some(first) = result.choices.first() {
        if !first.text.is_empty() {
            return first.text.clone();
        }
    }
    // Rule 2: (placeholder) if there were content arrays, join them here.
    // For M2 placeholder InvokeResult we only have `text`.

    // Rule 3: best-effort fallback
    String::new()
}
