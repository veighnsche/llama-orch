use crate::*;

/// Types of proof bundles supported in the monorepo.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TestType {
    Unit,
    Integration,
    Contract,
    Bdd,
    Determinism,
    Smoke,
    E2eHaiku,
}

impl TestType {
    pub fn as_dir(self) -> &'static str {
        match self {
            TestType::Unit => "unit",
            TestType::Integration => "integration",
            TestType::Contract => "contract",
            TestType::Bdd => "bdd",
            TestType::Determinism => "determinism",
            TestType::Smoke => "home-profile-smoke",
            TestType::E2eHaiku => "e2e-haiku",
        }
    }
}
