use std::env;

/// Resolve a short SHA (8 chars) from well-known CI environment variables.
pub fn resolve_sha8() -> Option<String> {
    let cand = env::var("GIT_SHA")
        .ok()
        .or_else(|| env::var("CI_COMMIT_SHA").ok())
        .or_else(|| env::var("GITHUB_SHA").ok());
    cand.map(|s| s.chars().take(8).collect())
}
