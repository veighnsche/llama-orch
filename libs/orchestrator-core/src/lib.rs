//! Core orchestrator library (pre-code).
//! Queueing, scheduling hooks, and core invariants. No HTTP or adapter traits here.
//!
//! Traceability (SPEC):
//! - OC-CORE-1001, OC-CORE-1002, OC-CORE-1004 (queue & admission invariants)
//! - OC-CORE-1010, OC-CORE-1011, OC-CORE-1012 (placement & readiness)
//! - OC-CORE-1030 (determinism invariants)
//! - OC-CORE-1040, OC-CORE-1041 (observability fields)

pub mod queue;
