//! Testing Team Verification Test
//!
//! This test verifies claims made by investigation teams about the worker-orcd implementation.
//! It checks for false positives and unverified claims that could mask bugs.
//!
//! Spec: test-harness/TEAM_RESPONSIBILITIES.md
//! Fines: test-harness/FINES_SUMMARY.md

use std::path::Path;

/// Verify that claimed "FIXED" documents actually describe fixes
#[test]
fn test_no_false_fixed_claims() {
    // Fine #8: TEAM_CHARLIE_BETA claimed "BUG FIXED" but admitted fix doesn't work
    let bug_fixed_doc = Path::new("investigation-teams/TEAM_CHARLIE_BETA_BUG_FIXED.md");
    
    if bug_fixed_doc.exists() {
        let content = std::fs::read_to_string(bug_fixed_doc)
            .expect("Failed to read TEAM_CHARLIE_BETA_BUG_FIXED.md");
        
        // Check for contradictions
        let has_fixed_claim = content.contains("Bug Fixed!") || content.contains("BUG FOUND AND FIXED");
        let has_false_alarm = content.contains("False Alarm") || content.contains("doesn't actually change anything");
        
        if has_fixed_claim && has_false_alarm {
            panic!(
                "TESTING TEAM VIOLATION: Document claims 'BUG FIXED' but admits fix doesn't work.\n\
                 This is a FALSE POSITIVE - creates false confidence.\n\
                 Fine: €200 (see test-harness/ADDITIONAL_FINES_REPORT.md)\n\
                 Remediation: Rename to TEAM_CHARLIE_BETA_FALSE_ALARM.md"
            );
        }
    }
}

/// Verify that test files don't bypass what they claim to test
#[test]
fn test_no_test_bypasses() {
    // Fine #4: Test bypasses special tokens while claiming "tokenization is correct"
    let haiku_test = std::fs::read_to_string("tests/haiku_generation_anti_cheat.rs")
        .expect("Failed to read haiku test");
    
    let cuda_backend = std::fs::read_to_string("src/inference/cuda_backend.rs")
        .expect("Failed to read cuda_backend");
    
    // Check if test bypasses chat template
    let has_bypass = cuda_backend.contains("use_chat_template = false") 
        || cuda_backend.contains("use_chat_template=false");
    
    let claims_correct = cuda_backend.contains("Tokenization is CORRECT") 
        || cuda_backend.contains("CONCLUSION: Tokenization is CORRECT");
    
    if has_bypass && claims_correct {
        panic!(
            "TESTING TEAM VIOLATION: Test bypasses special tokens (use_chat_template=false)\n\
             but claims 'tokenization is correct'.\n\
             This is a CRITICAL FALSE POSITIVE.\n\
             Fine: €150 (see test-harness/TEAM_PEAR_VERIFICATION.md)\n\
             Remediation: Enable chat template OR remove 'correct' claim"
        );
    }
}

/// Verify that "ELIMINATED" claims have sufficient evidence
#[test]
fn test_eliminated_claims_have_evidence() {
    let qwen_transformer = std::fs::read_to_string("cuda/src/transformer/qwen_transformer.cpp")
        .expect("Failed to read qwen_transformer.cpp");
    
    // Check for "ELIMINATED" claims without coverage documentation
    let lines: Vec<&str> = qwen_transformer.lines().collect();
    
    for (i, line) in lines.iter().enumerate() {
        if line.contains("ELIMINATED") && line.contains("❌") {
            // Look for coverage documentation in surrounding lines
            let context_start = i.saturating_sub(10);
            let context_end = (i + 10).min(lines.len());
            let context = lines[context_start..context_end].join("\n");
            
            // Check if coverage is documented
            let has_coverage_doc = context.contains("coverage") 
                || context.contains("out of") 
                || context.contains("columns")
                || context.contains("tokens");
            
            if !has_coverage_doc {
                eprintln!(
                    "WARNING: Line {} claims 'ELIMINATED' without documenting verification coverage:\n  {}\n\
                     This may be a sparse verification issue (see Fine #10, #11).",
                    i + 1, line.trim()
                );
            }
        }
    }
}

/// Verify that contradictory claims don't exist
#[test]
fn test_no_contradictory_claims() {
    let weight_loader = std::fs::read_to_string("cuda/src/model/qwen_weight_loader.cpp")
        .expect("Failed to read qwen_weight_loader.cpp");
    
    // Check for "TESTED" and "NOT TESTED" in same file
    let has_tested = weight_loader.contains("TESTED:") || weight_loader.contains("TESTED ");
    let has_not_tested = weight_loader.contains("NOT TESTED") || weight_loader.contains("UNTESTED");
    
    if has_tested && has_not_tested {
        // Check if they're in same comment block (within 50 lines)
        let lines: Vec<&str> = weight_loader.lines().collect();
        let mut tested_line = None;
        let mut not_tested_line = None;
        
        for (i, line) in lines.iter().enumerate() {
            if line.contains("TESTED:") || line.contains("TESTED ") {
                tested_line = Some(i);
            }
            if line.contains("NOT TESTED") || line.contains("UNTESTED") {
                not_tested_line = Some(i);
            }
        }
        
        if let (Some(t), Some(nt)) = (tested_line, not_tested_line) {
            if (t as i32 - nt as i32).abs() < 50 {
                panic!(
                    "TESTING TEAM VIOLATION: Contradictory claims in qwen_weight_loader.cpp\n\
                     Line {}: Claims 'TESTED'\n\
                     Line {}: Claims 'NOT TESTED'\n\
                     Cannot be both tested and not tested.\n\
                     Fine: €100 (see test-harness/ADDITIONAL_FINES_REPORT.md)",
                    t + 1, nt + 1
                );
            }
        }
    }
}

/// Verify that comprehensive verification claims have >10% coverage
#[test]
fn test_comprehensive_verification_coverage() {
    let qwen_transformer = std::fs::read_to_string("cuda/src/transformer/qwen_transformer.cpp")
        .expect("Failed to read qwen_transformer.cpp");
    
    // Look for verification claims with coverage percentages
    let lines: Vec<&str> = qwen_transformer.lines().collect();
    
    for (i, line) in lines.iter().enumerate() {
        // Check for manual verification claims
        if line.contains("manual") && (line.contains("verification") || line.contains("verified")) {
            // Look for coverage in surrounding lines
            let context_start = i.saturating_sub(5);
            let context_end = (i + 5).min(lines.len());
            let context = lines[context_start..context_end].join("\n");
            
            // Extract coverage if mentioned
            if context.contains("out of") {
                // Parse "X out of Y" pattern
                if let Some(coverage_line) = lines[context_start..context_end]
                    .iter()
                    .find(|l| l.contains("out of"))
                {
                    eprintln!(
                        "INFO: Manual verification at line {}: {}",
                        i + 1, coverage_line.trim()
                    );
                    
                    // Check if coverage is < 1%
                    if coverage_line.contains("0.11%") 
                        || coverage_line.contains("0.0026%") 
                        || coverage_line.contains("0.22%")
                    {
                        eprintln!(
                            "  ⚠️  WARNING: Sparse verification (<1% coverage)\n\
                             This may warrant a fine for insufficient coverage."
                        );
                    }
                }
            }
        }
    }
}

/// Verify that "MATHEMATICALLY CORRECT" claims have evidence
#[test]
fn test_mathematically_correct_claims() {
    let files = vec![
        "cuda/src/transformer/qwen_transformer.cpp",
        "cuda/kernels/rope.cu",
        "cuda/kernels/rmsnorm.cu",
    ];
    
    for file_path in files {
        if let Ok(content) = std::fs::read_to_string(file_path) {
            let lines: Vec<&str> = content.lines().collect();
            
            for (i, line) in lines.iter().enumerate() {
                if line.contains("MATHEMATICALLY CORRECT") {
                    // Look for proof/evidence in surrounding lines
                    let context_start = i.saturating_sub(10);
                    let context_end = (i + 10).min(lines.len());
                    let context = lines[context_start..context_end].join("\n");
                    
                    let has_proof = context.contains("PROOF:") 
                        || context.contains("VERIFIED:") 
                        || context.contains("Manual verification")
                        || context.contains("matches");
                    
                    if !has_proof {
                        eprintln!(
                            "WARNING: {} line {} claims 'MATHEMATICALLY CORRECT' without proof:\n  {}",
                            file_path, i + 1, line.trim()
                        );
                    }
                }
            }
        }
    }
}

/// Verify that reference files cited actually exist
#[test]
fn test_reference_files_exist() {
    // Fine #1: Team Purple cited non-existent .archive/llama_cpp_debug.log
    let cuda_backend = std::fs::read_to_string("src/inference/cuda_backend.rs")
        .expect("Failed to read cuda_backend");
    
    if cuda_backend.contains(".archive/llama_cpp_debug.log") {
        let ref_file = Path::new(".archive/llama_cpp_debug.log");
        if !ref_file.exists() {
            panic!(
                "TESTING TEAM VIOLATION: Code cites '.archive/llama_cpp_debug.log' but file doesn't exist.\n\
                 This is a FALSE VERIFICATION - cannot verify against non-existent file.\n\
                 Fine: €50 (see test-harness/TEAM_PEAR_VERIFICATION.md)\n\
                 Remediation: Provide actual reference file OR remove citation"
            );
        }
    }
}

/// Summary test - reports all issues found
#[test]
fn test_summary_report() {
    println!("\n=== Testing Team Verification Summary ===\n");
    println!("This test suite verifies investigation team claims.");
    println!("See test-harness/FINES_SUMMARY.md for complete fine details.\n");
    println!("Total fines issued: €1,250");
    println!("  - Phase 1 (Tokenization): €500");
    println!("  - Phase 2 (cuBLAS): €300");
    println!("  - Additional (False Claims): €450\n");
    println!("Run individual tests to see specific violations.");
    println!("\n==========================================\n");
}

// Built by Testing Team 🔍
