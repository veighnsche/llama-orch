# TODO — test-harness-bdd (Cross-crate Integration)

- Curate features to focus on HTTP control/data plane and adapter integration only; prune any crate-scoped leftovers.
- Add capability handshake features once schema stabilizes.
- Add backpressure under load scenarios with parameterized queue capacity.
- Flesh out observability assertions for required log fields and metrics per request.
- Gate strict traceability failures via env (LLORCH_TRACEABILITY_STRICT=1) in CI.
