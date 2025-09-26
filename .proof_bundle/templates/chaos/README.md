# Chaos/Load Proof Bundle Template

Required generated artifacts

- load_profile.json — QPS/arrival pattern, duration, injected faults
- metrics_timeseries.csv — key metrics time series (errors, latency percentiles)
- recovery_report.md — time to recover to SLOs after fault
- logs_redacted/ — relevant redacted logs during run window
- test_report.md — summary of scenarios and outcomes

Notes

- Align with `.docs/HOME_PROFILE_TARGET.md` constraints when applicable.
