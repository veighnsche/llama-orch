#[napi(object)]
pub struct ChoiceNapi { pub text: String }

#[napi(object)]
pub struct UsageNapi {
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
}

#[napi(object)]
pub struct InvokeResultNapi {
    pub choices: Vec<ChoiceNapi>,
    pub usage: Option<UsageNapi>,
}

// Conversions
impl From<InvokeResultNapi> for llama_orch_utils::llm::invoke::InvokeResult {
    fn from(value: InvokeResultNapi) -> Self {
        llama_orch_utils::llm::invoke::InvokeResult {
            choices: value.choices.into_iter().map(|c| llama_orch_utils::llm::invoke::Choice { text: c.text }).collect(),
            usage: value.usage.map(|u| llama_orch_utils::llm::invoke::Usage {
                prompt_tokens: u.prompt_tokens.map(|v| v as i32),
                completion_tokens: u.completion_tokens.map(|v| v as i32),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_rules() {
        // First choice non-empty
        let r1 = InvokeResultNapi { choices: vec![ChoiceNapi { text: "abc".into() }], usage: None };
        let out1 = llama_orch_utils::orch::response_extractor::run(&r1.into());
        assert_eq!(out1, "abc");

        // First empty, second non-empty
        let r2 = InvokeResultNapi { choices: vec![ChoiceNapi { text: "".into() }, ChoiceNapi { text: "xyz".into() }], usage: None };
        let out2 = llama_orch_utils::orch::response_extractor::run(&r2.into());
        assert_eq!(out2, "xyz");

        // All empty -> ""
        let r3 = InvokeResultNapi { choices: vec![ChoiceNapi { text: "".into() }], usage: None };
        let out3 = llama_orch_utils::orch::response_extractor::run(&r3.into());
        assert_eq!(out3, "");
    }
}
