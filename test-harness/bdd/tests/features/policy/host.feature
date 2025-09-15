Feature: Policy Host ABI and Safety
  # Traceability: OC-POLICY-4001..4020
  Scenario: WASI ABI, determinism, sandboxing and telemetry
    Given a policy host
    Then the default plugin ABI is WASI
    And functions are pure and deterministic over explicit snapshots
    And ABI versioning is explicit and bumps MAJOR on breaking changes
    And plugins run in a sandbox with no filesystem or network by default
    And host bounds CPU time and memory per invocation
    And host logs plugin id version decision and latency
