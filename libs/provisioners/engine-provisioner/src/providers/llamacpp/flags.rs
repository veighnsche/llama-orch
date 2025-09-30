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
