/// Normalize llama.cpp CLI flags.
/// - Map legacy options to current llama.cpp conventions and enforce CPU/GPU consistency.
///   - "--ngl N", "-ngl N", or "--gpu-layers N" -> "--n-gpu-layers N"
///   - If CPU-only, force "--n-gpu-layers 0" and drop any previous GPU layer flags.
pub fn normalize_llamacpp_flags(flags: &[String], gpu_enabled: bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < flags.len() {
        let f = &flags[i];
        let next = flags.get(i + 1);
        let is_pair = |s: &str| s == "--ngl" || s == "-ngl" || s == "--gpu-layers" || s == "--n-gpu-layers";
        if is_pair(f) {
            // Consume this flag and its value
            if let Some(val) = next {
                if gpu_enabled {
                    out.push("--n-gpu-layers".to_string());
                    out.push(val.clone());
                }
                i += 2;
                continue;
            } else {
                // Malformed pair; skip it
                i += 1;
                continue;
            }
        }
        // passthrough any other flag
        out.push(f.clone());
        i += 1;
    }
    if !gpu_enabled {
        // Remove any accidental n-gpu-layers set and enforce 0
        let mut cleaned: Vec<String> = Vec::new();
        let mut j = 0usize;
        while j < out.len() {
            let s = &out[j];
            if s == "--n-gpu-layers" {
                // drop this and its value
                j += 2;
                continue;
            }
            cleaned.push(s.clone());
            j += 1;
        }
        cleaned.push("--n-gpu-layers".to_string());
        cleaned.push("0".to_string());
        return cleaned;
    }
    // If GPU enabled and no layer flag was provided, leave as-is (llama.cpp will choose default)
    out
}

#[cfg(test)]
mod tests {
    use super::normalize_llamacpp_flags;

    fn svec(xs: &[&str]) -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn gpu_enabled_normalizes_all_pair_variants() {
        let flags = svec(&["--ngl", "35", "-ngl", "12", "--gpu-layers", "7", "--n-gpu-layers", "3", "--other", "x"]);
        let out = normalize_llamacpp_flags(&flags, true);
        // Each pair should be rewritten to --n-gpu-layers N and others passthrough, order preserved for non-pairs
        assert_eq!(out, svec(&[
            "--n-gpu-layers", "35",
            "--n-gpu-layers", "12",
            "--n-gpu-layers", "7",
            "--n-gpu-layers", "3",
            "--other", "x",
        ]));
    }

    #[test]
    fn gpu_enabled_ignores_malformed_pair_without_value() {
        let flags = svec(&["--ngl", "32", "--gpu-layers"]); // last is missing value
        let out = normalize_llamacpp_flags(&flags, true);
        assert_eq!(out, svec(&["--n-gpu-layers", "32"]));
    }

    #[test]
    fn cpu_mode_forces_zero_layers_and_removes_existing_specs() {
        let flags = svec(&["--n-gpu-layers", "5", "--other", "abc", "--ngl", "12"]);
        let out = normalize_llamacpp_flags(&flags, false);
        // Any preexisting layer settings removed; enforced 0 appended at end
        assert_eq!(out, svec(&["--other", "abc", "--n-gpu-layers", "0"]));
    }

    #[test]
    fn passthrough_unrelated_flags_in_gpu_and_cpu_modes() {
        let flags = svec(&["--foo", "bar", "--baz"]);
        let out_gpu = normalize_llamacpp_flags(&flags, true);
        assert_eq!(out_gpu, svec(&["--foo", "bar", "--baz"]));
        let out_cpu = normalize_llamacpp_flags(&flags, false);
        assert_eq!(out_cpu, svec(&["--foo", "bar", "--baz", "--n-gpu-layers", "0"]));
    }
}

