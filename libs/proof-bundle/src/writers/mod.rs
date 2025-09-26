use crate::fs::ProofBundle;
use anyhow::Result;
use serde::Serialize;

pub struct Writers<'a> {
    pub(crate) pb: &'a ProofBundle,
}

impl<'a> Writers<'a> {
    pub fn markdown(&self) -> MarkdownWriter<'a> {
        MarkdownWriter { pb: self.pb }
    }
    pub fn json(&self) -> JsonWriter<'a> {
        JsonWriter { pb: self.pb }
    }
    pub fn ndjson(&self) -> NdjsonWriter<'a> {
        NdjsonWriter { pb: self.pb }
    }
    pub fn meta(&self) -> MetaWriter<'a> {
        MetaWriter { pb: self.pb }
    }
}

pub struct MarkdownWriter<'a> {
    pub(crate) pb: &'a ProofBundle,
}
impl<'a> MarkdownWriter<'a> {
    pub fn write<N: AsRef<str>>(&self, name: N, body: &str) -> Result<()> {
        self.pb.write_markdown(name, body)
    }
}

pub struct JsonWriter<'a> {
    pub(crate) pb: &'a ProofBundle,
}
impl<'a> JsonWriter<'a> {
    pub fn write<T: Serialize, N: AsRef<str>>(&self, base: N, value: &T) -> Result<()> {
        self.pb.write_json(base, value)
    }
    pub fn write_with_meta<T: Serialize, N: AsRef<str>>(&self, base: N, value: &T) -> Result<()> {
        self.pb.write_json_with_meta(base, value)
    }
}

pub struct NdjsonWriter<'a> {
    pub(crate) pb: &'a ProofBundle,
}
impl<'a> NdjsonWriter<'a> {
    pub fn append<T: Serialize, N: AsRef<str>>(&self, name: N, value: &T) -> Result<()> {
        self.pb.append_ndjson(name, value)
    }
    pub fn ensure_meta<N: AsRef<str>>(&self, name: N) -> Result<()> {
        self.pb.append_ndjson_autogen_meta(name)
    }
}

pub struct MetaWriter<'a> {
    pub(crate) pb: &'a ProofBundle,
}
impl<'a> MetaWriter<'a> {
    pub fn write_for<N: AsRef<str>>(&self, name: N) -> Result<()> {
        self.pb.write_meta_sibling(name)
    }
}
