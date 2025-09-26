# Determinism Suite Proof Bundle Template

Required generated artifacts

- pairs/ — side-by-side token stream outputs for identical inputs across replicas
- diffs/ — computed diffs highlighting mismatches (byte/token level)
- run_config.json — engine_version, sampler_profile_version, model_digest, seeds
- timing.csv — (optional) latency per token for reference
- test_report.md — summary, counts of matching/mismatching pairs

Notes

- See `/.specs/00_llama-orch.md` §5 (Determinism suite requirement).
- Enforce REQUIRE_REAL_LLAMA if applicable.
