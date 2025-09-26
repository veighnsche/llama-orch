use std::fs;

#[test]
fn seeds_file_has_64_lines() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let seeds_path = root.join("seeds.txt");
    let contents = fs::read_to_string(seeds_path).expect("seeds.txt");
    let count = contents.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).count();
    assert_eq!(count, 64, "expected 64 seeded prompts");
}
