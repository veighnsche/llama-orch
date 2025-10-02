//! Metadata extraction from source code

pub mod parser;
pub mod annotations;
pub mod cache;

pub use parser::extract_metadata;
pub use cache::MetadataCache;
