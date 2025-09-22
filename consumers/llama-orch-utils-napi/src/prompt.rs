// N-API wrappers for prompt module (M2 DRAFT)
// Source representation chosen:
// - Discriminated object with `kind: "Text" | "Lines" | "File"`.
// - For `Text`, use field `text: string`.
// - For `Lines`, use field `lines: string[]`.
// - For `File`, use field `path: string`.
// Exactly one payload field must be present per kind. Conversions are total and
// return napi::Error on shape mismatches (no unwrap/expect).

// Avoid importing napi::Result alias to keep error types unambiguous in TryFrom impls.

#[napi(object)]
pub struct SourceNapi {
    pub kind: String,
    pub text: Option<String>,
    pub lines: Option<Vec<String>>,
    pub path: Option<String>,
}

#[napi(object)]
pub struct MessageInNapi {
    pub role: String,
    pub source: SourceNapi,
    pub dedent: bool,
}

#[napi(object)]
pub struct MessageNapi {
    pub role: String,
    pub content: String,
}

#[napi(object)]
pub struct ThreadItemNapi {
    pub role: String,
    pub source: SourceNapi,
    pub dedent: bool,
}

#[napi(object)]
pub struct ThreadInNapi {
    pub items: Vec<ThreadItemNapi>,
}

#[napi(object)]
pub struct ThreadOutNapi {
    pub messages: Vec<MessageNapi>,
}

// ---------- Conversions ----------

impl TryFrom<SourceNapi> for llama_orch_utils::prompt::message::Source {
    type Error = String;
    fn try_from(value: SourceNapi) -> std::result::Result<Self, Self::Error> {
        match value.kind.as_str() {
            "Text" => {
                let s = value
                    .text
                    .ok_or_else(|| "SourceNapi(Text): missing 'text'".to_string())?;
                Ok(llama_orch_utils::prompt::message::Source::Text(s))
            }
            "Lines" => {
                let v = value
                    .lines
                    .ok_or_else(|| "SourceNapi(Lines): missing 'lines'".to_string())?;
                Ok(llama_orch_utils::prompt::message::Source::Lines(v))
            }
            "File" => {
                let p = value
                    .path
                    .ok_or_else(|| "SourceNapi(File): missing 'path'".to_string())?;
                Ok(llama_orch_utils::prompt::message::Source::File(p))
            }
            other => Err(format!(
                "SourceNapi: unknown kind '{}'; expected Text|Lines|File",
                other
            )),
        }
    }
}

impl From<llama_orch_utils::prompt::message::Message> for MessageNapi {
    fn from(value: llama_orch_utils::prompt::message::Message) -> Self {
        MessageNapi { role: value.role, content: value.content }
    }
}

impl TryFrom<MessageInNapi> for llama_orch_utils::prompt::message::MessageIn {
    type Error = String;
    fn try_from(value: MessageInNapi) -> std::result::Result<Self, Self::Error> {
        let src = llama_orch_utils::prompt::message::Source::try_from(value.source)?;
        Ok(llama_orch_utils::prompt::message::MessageIn { role: value.role, source: src, dedent: value.dedent })
    }
}

impl TryFrom<ThreadItemNapi> for llama_orch_utils::prompt::thread::ThreadItem {
    type Error = String;
    fn try_from(value: ThreadItemNapi) -> std::result::Result<Self, Self::Error> {
        let src = llama_orch_utils::prompt::message::Source::try_from(value.source)?;
        Ok(llama_orch_utils::prompt::thread::ThreadItem { role: value.role, source: src, dedent: value.dedent })
    }
}

impl TryFrom<ThreadInNapi> for llama_orch_utils::prompt::thread::ThreadIn {
    type Error = String;
    fn try_from(value: ThreadInNapi) -> std::result::Result<Self, Self::Error> {
        let mut items = Vec::with_capacity(value.items.len());
        for it in value.items.into_iter() {
            items.push(llama_orch_utils::prompt::thread::ThreadItem::try_from(it)?);
        }
        Ok(llama_orch_utils::prompt::thread::ThreadIn { items })
    }
}

impl From<llama_orch_utils::prompt::thread::ThreadOut> for ThreadOutNapi {
    fn from(value: llama_orch_utils::prompt::thread::ThreadOut) -> Self {
        ThreadOutNapi {
            messages: value
                .messages
                .into_iter()
                .map(|m| MessageNapi { role: m.role, content: m.content })
                .collect(),
        }
    }
}

// ---------- Tests (conversions only) ----------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_text_and_lines_convert() {
        let s_text = SourceNapi { kind: "Text".into(), text: Some("hi".into()), lines: None, path: None };
        let core_text = llama_orch_utils::prompt::message::Source::try_from(s_text).unwrap();
        match core_text { llama_orch_utils::prompt::message::Source::Text(s) => assert_eq!(s, "hi"), _ => panic!("wrong variant") }

        let s_lines = SourceNapi { kind: "Lines".into(), text: None, lines: Some(vec!["a".into(), "b".into()]), path: None };
        let core_lines = llama_orch_utils::prompt::message::Source::try_from(s_lines).unwrap();
        match core_lines { llama_orch_utils::prompt::message::Source::Lines(v) => assert_eq!(v, vec!["a", "b"]), _ => panic!("wrong variant") }
    }

    #[test]
    fn message_roundtrip_shapes() {
        let min = MessageInNapi { role: "user".into(), source: SourceNapi { kind: "Text".into(), text: Some("hello".into()), lines: None, path: None }, dedent: false };
        let core_in = llama_orch_utils::prompt::message::MessageIn::try_from(min).unwrap();
        assert_eq!(core_in.role, "user");
        match core_in.source { llama_orch_utils::prompt::message::Source::Text(s) => assert_eq!(s, "hello"), _ => panic!("expected Text") }
        assert!(!core_in.dedent);

        let core_msg = llama_orch_utils::prompt::message::Message { role: "assistant".into(), content: "ok".into() };
        let msg_napi: MessageNapi = core_msg.into();
        assert_eq!(msg_napi.role, "assistant");
        assert_eq!(msg_napi.content, "ok");
    }

    #[test]
    fn thread_mapping_preserves_order() {
        let t_in = ThreadInNapi {
            items: vec![
                ThreadItemNapi { role: "system".into(), source: SourceNapi { kind: "Text".into(), text: Some("a".into()), lines: None, path: None }, dedent: false },
                ThreadItemNapi { role: "user".into(), source: SourceNapi { kind: "Lines".into(), text: None, lines: Some(vec!["b".into(), "c".into()]), path: None }, dedent: true },
            ],
        };
        let core_tin = llama_orch_utils::prompt::thread::ThreadIn::try_from(t_in).unwrap();
        assert_eq!(core_tin.items.len(), 2);

        let core_tout = llama_orch_utils::prompt::thread::ThreadOut {
            messages: vec![
                llama_orch_utils::prompt::message::Message { role: "system".into(), content: "a".into() },
                llama_orch_utils::prompt::message::Message { role: "user".into(), content: "b\nc".into() },
            ],
        };
        let tout_napi: ThreadOutNapi = core_tout.into();
        assert_eq!(tout_napi.messages.len(), 2);
        assert_eq!(tout_napi.messages[0].role, "system");
        assert_eq!(tout_napi.messages[0].content, "a");
        assert_eq!(tout_napi.messages[1].content, "b\nc");
    }
}
