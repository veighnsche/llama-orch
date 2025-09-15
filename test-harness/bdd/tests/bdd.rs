use regex::Regex;
use std::fs;
use std::path::PathBuf;

fn feature_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/features")
}

#[test]
fn features_have_no_undefined_or_ambiguous_steps() {
    let dir = feature_dir();
    assert!(dir.exists(), "features dir missing: {}", dir.display());

    // Define step regexes (add more over time)
    let steps: Vec<Regex> = vec![
        Regex::new(r"^noop$").unwrap(),
        Regex::new(r"^nothing happens$").unwrap(),
        Regex::new(r"^it passes$").unwrap(),
    ];

    for entry in fs::read_dir(&dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            let text = fs::read_to_string(entry.path()).unwrap();
            for line in text.lines() {
                let line = line.trim();
                if line.starts_with("Given ") {
                    let s = line.trim_start_matches("Given ").trim();
                    check_step(s, &steps, &entry.path());
                } else if line.starts_with("When ") {
                    let s = line.trim_start_matches("When ").trim();
                    check_step(s, &steps, &entry.path());
                } else if line.starts_with("Then ") {
                    let s = line.trim_start_matches("Then ").trim();
                    check_step(s, &steps, &entry.path());
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
    assert!(
        matches > 0,
        "undefined step '{}' in {}",
        step,
        file.display()
    );
    assert!(
        matches == 1,
        "ambiguous step '{}' in {} ({} matches)",
        step,
        file.display(),
        matches
    );
}
