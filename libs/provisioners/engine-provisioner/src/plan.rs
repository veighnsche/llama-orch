use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub kind: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Plan {
    pub pool_id: String,
    pub steps: Vec<PlanStep>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip() {
        let plan = Plan {
            pool_id: "poolA".into(),
            steps: vec![
                PlanStep {
                    kind: "preflight-tools".into(),
                    detail: "allow_package_installs=None".into(),
                },
                PlanStep { kind: "run".into(), detail: "ports=[8080]".into() },
            ],
        };
        let json = serde_json::to_string(&plan).expect("serialize");
        let back: Plan = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.pool_id, "poolA");
        assert_eq!(back.steps.len(), 2);
        assert_eq!(back.steps[0].kind, "preflight-tools");
        assert_eq!(back.steps[1].kind, "run");
    }
}
