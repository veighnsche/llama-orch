use regex::Regex;
use std::fs;
use std::path::PathBuf;
use test_harness_bdd::steps;
use walkdir::WalkDir;

fn feature_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/features")
}

#[test]
fn features_have_no_undefined_or_ambiguous_steps() {
    let dir = feature_dir();
    assert!(dir.exists(), "features dir missing: {}", dir.display());

    // Define step regexes (add more over time via registry)
    let steps: Vec<Regex> = steps::registry();

    for entry in WalkDir::new(&dir).into_iter().filter_map(Result::ok) {
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|s| s.to_str()) == Some("feature")
        {
            let path = entry.into_path();
            let text = fs::read_to_string(&path).unwrap();
            for line in text.lines() {
                let line = line.trim();
                if line.starts_with("Given ") {
                    let s = line.trim_start_matches("Given ").trim();
                    check_step(s, &steps, &path);
                } else if line.starts_with("When ") {
                    let s = line.trim_start_matches("When ").trim();
                    check_step(s, &steps, &path);
                } else if line.starts_with("Then ") {
                    let s = line.trim_start_matches("Then ").trim();
                    check_step(s, &steps, &path);
                } else if line.starts_with("And ") {
                    let s = line.trim_start_matches("And ").trim();
                    check_step(s, &steps, &path);
                }
            }
        }
    }
}

fn check_step(step: &str, steps: &[Regex], file: &std::path::Path) {
    let mut matches = 0;
    for re in steps {
        if re.is_match(step) {
            matches += 1;
        }
    }
    assert!(matches > 0, "undefined step '{}' in {}", step, file.display());
    assert!(matches == 1, "ambiguous step '{}' in {} ({} matches)", step, file.display(), matches);
}
