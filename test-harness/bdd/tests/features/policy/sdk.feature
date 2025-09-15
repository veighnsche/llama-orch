Feature: Policy SDK stability and safety
  # Traceability: OC-POLICY-SDK-4101..4110
  Scenario: Semver stability and safety defaults
    Given a policy SDK
    Then public SDK functions are semver-stable within a MAJOR
    And breaking changes include a migration note and version bump
    And SDK performs no network or filesystem I/O by default
