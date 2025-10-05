// GGUF Merges Parser (LT-008)
//
// Parses BPE merge rules from GGUF metadata to enable byte-level BPE encoding.
//
// Spec: M0-W-1362

use super::error::MergeError;
use std::collections::BTreeMap;

/// BPE merge pair (left token + right token)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MergePair {
    pub left: String,
    pub right: String,
}

impl MergePair {
    pub fn new(left: String, right: String) -> Self {
        Self { left, right }
    }
}

/// BPE merge table with priorities
#[derive(Debug, Clone)]
pub struct MergeTable {
    /// Merge pair → priority (lower priority = earlier merge)
    pub merge_priority: BTreeMap<MergePair, u32>,

    /// Total number of merges
    pub merge_count: u32,
}

impl MergeTable {
    /// Create new merge table from merge lines
    pub fn new(merge_lines: Vec<String>) -> Result<Self, MergeError> {
        if merge_lines.is_empty() {
            return Err(MergeError::InvalidCount { count: 0 });
        }

        let mut merge_priority = BTreeMap::new();

        for (priority, line) in merge_lines.iter().enumerate() {
            let pair = Self::parse_merge_line(line)?;
            merge_priority.insert(pair, priority as u32);
        }

        Ok(Self { merge_priority, merge_count: merge_lines.len() as u32 })
    }

    /// Parse a single merge line: "left right" → MergePair
    fn parse_merge_line(line: &str) -> Result<MergePair, MergeError> {
        let parts: Vec<&str> = line.split(' ').collect();

        if parts.len() != 2 {
            return Err(MergeError::MalformedLine { line: line.to_string() });
        }

        Ok(MergePair::new(parts[0].to_string(), parts[1].to_string()))
    }

    /// Get merge priority for a pair (returns None if pair not in table)
    pub fn get_priority(&self, left: &str, right: &str) -> Option<u32> {
        let pair = MergePair::new(left.to_string(), right.to_string());
        self.merge_priority.get(&pair).copied()
    }

    /// Check if a merge pair exists
    pub fn contains_pair(&self, left: &str, right: &str) -> bool {
        self.get_priority(left, right).is_some()
    }
}

/// Merges parser for GGUF metadata
pub struct MergesParser;

impl MergesParser {
    /// Parse BPE merges from GGUF metadata
    ///
    /// Extracts `tokenizer.ggml.merges` array (merge lines in format "left right")
    pub fn parse_from_metadata(merge_lines: Vec<String>) -> Result<MergeTable, MergeError> {
        if merge_lines.is_empty() {
            return Err(MergeError::MissingMetadata);
        }

        tracing::info!("Parsing BPE merges: {} merge rules", merge_lines.len());

        MergeTable::new(merge_lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_merges() -> MergeTable {
        let merge_lines = vec![
            "Ġ t".to_string(), // priority 0
            "h e".to_string(), // priority 1
            "l l".to_string(), // priority 2
            "o Ġ".to_string(), // priority 3
        ];

        MergeTable::new(merge_lines).unwrap()
    }

    #[test]
    fn test_merge_table_creation() {
        let merges = create_test_merges();

        assert_eq!(merges.merge_count, 4);
        assert_eq!(merges.merge_priority.len(), 4);
    }

    #[test]
    fn test_merge_priority_lookup() {
        let merges = create_test_merges();

        assert_eq!(merges.get_priority("Ġ", "t"), Some(0));
        assert_eq!(merges.get_priority("h", "e"), Some(1));
        assert_eq!(merges.get_priority("l", "l"), Some(2));
        assert_eq!(merges.get_priority("o", "Ġ"), Some(3));
        assert_eq!(merges.get_priority("x", "y"), None);
    }

    #[test]
    fn test_contains_pair() {
        let merges = create_test_merges();

        assert!(merges.contains_pair("Ġ", "t"));
        assert!(merges.contains_pair("h", "e"));
        assert!(!merges.contains_pair("x", "y"));
    }

    #[test]
    fn test_parse_merge_line() {
        let pair = MergeTable::parse_merge_line("Ġ t").unwrap();
        assert_eq!(pair.left, "Ġ");
        assert_eq!(pair.right, "t");

        let pair = MergeTable::parse_merge_line("hello world").unwrap();
        assert_eq!(pair.left, "hello");
        assert_eq!(pair.right, "world");
    }

    #[test]
    fn test_malformed_merge_line() {
        let result = MergeTable::parse_merge_line("single");
        assert!(matches!(result, Err(MergeError::MalformedLine { .. })));

        let result = MergeTable::parse_merge_line("too many tokens");
        assert!(matches!(result, Err(MergeError::MalformedLine { .. })));
    }

    #[test]
    fn test_empty_merge_table() {
        let result = MergeTable::new(vec![]);
        assert!(matches!(result, Err(MergeError::InvalidCount { .. })));
    }

    #[test]
    fn test_merges_parser() {
        let merge_lines = vec!["a b".to_string(), "c d".to_string()];
        let merges = MergesParser::parse_from_metadata(merge_lines).unwrap();

        assert_eq!(merges.merge_count, 2);
        assert_eq!(merges.get_priority("a", "b"), Some(0));
        assert_eq!(merges.get_priority("c", "d"), Some(1));
    }

    #[test]
    fn test_byte_level_bpe_characters() {
        let merge_lines = vec![
            "Ġ t".to_string(), // space + t
            "Ċ n".to_string(), // newline + n
        ];

        let merges = MergeTable::new(merge_lines).unwrap();

        assert_eq!(merges.get_priority("Ġ", "t"), Some(0));
        assert_eq!(merges.get_priority("Ċ", "n"), Some(1));
    }

    #[test]
    fn test_merge_priority_ordering() {
        let merge_lines =
            vec!["first merge".to_string(), "second merge".to_string(), "third merge".to_string()];

        let merges = MergeTable::new(merge_lines).unwrap();

        // First merge has lowest priority (0)
        assert_eq!(merges.get_priority("first", "merge"), Some(0));
        assert_eq!(merges.get_priority("second", "merge"), Some(1));
        assert_eq!(merges.get_priority("third", "merge"), Some(2));
    }
}
