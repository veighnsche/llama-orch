//! Metadata extraction from source code

#[cfg(feature = "metadata-extraction")]
pub mod parser;
#[cfg(feature = "metadata-extraction")]
pub mod annotations;
#[cfg(feature = "metadata-extraction")]
pub mod cache;

#[cfg(feature = "metadata-extraction")]
pub use parser::extract_metadata;
#[cfg(feature = "metadata-extraction")]
pub use cache::MetadataCache;

// Fallback stub when feature is disabled
#[cfg(not(feature = "metadata-extraction"))]
mod stub {
    use std::collections::HashMap;
    use crate::core::TestMetadata;
    use crate::Result;

    pub fn extract_metadata(_targets: &[crate::discovery::TestTarget]) -> Result<HashMap<String, TestMetadata>> {
        Ok(HashMap::new())
    }

    #[derive(Default)]
    pub struct MetadataCache;
}

#[cfg(not(feature = "metadata-extraction"))]
pub use stub::{extract_metadata, MetadataCache};
