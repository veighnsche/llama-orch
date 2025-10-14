// Placeholder module for applet prompt/thread.
use std::io;

use crate::prompt::message::{self, Message, Source};
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadItem {
    pub role: String,
    pub source: Source,
    pub dedent: bool,
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadIn {
    pub items: Vec<ThreadItem>,
}

#[cfg_attr(feature = "ts-types", derive(ts_rs::TS))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadOut {
    pub messages: Vec<Message>,
}

pub fn run(input: ThreadIn) -> io::Result<ThreadOut> {
    let mut out = Vec::with_capacity(input.items.len());
    for it in input.items.into_iter() {
        let msg = message::run(message::MessageIn {
            role: it.role,
            source: it.source,
            dedent: it.dedent,
        })?;
        out.push(msg);
    }
    Ok(ThreadOut { messages: out })
}
