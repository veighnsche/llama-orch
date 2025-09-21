// Placeholder module for applet prompt/thread.
use std::io;

use super::message::{self, Message, Source};

#[derive(Debug, Clone)]
pub struct ThreadItem {
    pub role: String,
    pub source: Source,
    pub dedent: bool,
}

#[derive(Debug, Clone)]
pub struct ThreadIn {
    pub items: Vec<ThreadItem>,
}

#[derive(Debug, Clone)]
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
