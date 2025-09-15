use openapiv3::{OpenAPI, Operation, ReferenceOr};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub fn openapi_summary(repo_root: &Path, refs: &[String]) -> String {
    let mut tags: BTreeSet<String> = BTreeSet::new();
    let mut op_ids: Vec<String> = Vec::new();
    for rel in refs {
        let abs = repo_root.join(rel);
        let Ok(txt) = fs::read_to_string(&abs) else {
            continue;
        };
        let Ok(doc) = serde_yaml::from_str::<OpenAPI>(&txt) else {
            continue;
        };
        for (_p, item) in doc.paths.paths {
            if let ReferenceOr::Item(pi) = item {
                collect_ops(&pi.get, &mut tags, &mut op_ids);
                collect_ops(&pi.put, &mut tags, &mut op_ids);
                collect_ops(&pi.post, &mut tags, &mut op_ids);
                collect_ops(&pi.delete, &mut tags, &mut op_ids);
                collect_ops(&pi.options, &mut tags, &mut op_ids);
                collect_ops(&pi.head, &mut tags, &mut op_ids);
                collect_ops(&pi.patch, &mut tags, &mut op_ids);
                collect_ops(&pi.trace, &mut tags, &mut op_ids);
            }
        }
    }
    op_ids.sort();
    op_ids.dedup();
    let mut out = String::new();
    if !op_ids.is_empty() {
        out.push_str(&format!("- OpenAPI operations: {}\n", op_ids.len()));
        let sample: Vec<String> = op_ids.iter().take(5).cloned().collect();
        if !sample.is_empty() {
            out.push_str(&format!("  - examples: {}\n", sample.join(", ")));
        }
    }
    if !tags.is_empty() {
        let tags_list: Vec<String> = tags.into_iter().collect();
        out.push_str(&format!("- OpenAPI tags: {}\n", tags_list.join(", ")));
    }
    out
}

fn collect_ops(op: &Option<Operation>, tags: &mut BTreeSet<String>, op_ids: &mut Vec<String>) {
    if let Some(op) = op {
        for t in &op.tags {
            tags.insert(t.clone());
        }
        if let Some(id) = &op.operation_id {
            op_ids.push(id.clone());
        }
    }
}
