// TEAM-252: State machine tests
// Purpose: Test all valid state transitions systematically

use crate::integration::harness::{TestHarness, ProcessState, SystemState};
use crate::integration::assertions::*;
use std::time::Duration;

/// All possible state transitions
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub initial: SystemState,
    pub command: Vec<String>,
    pub expected: SystemState,
    pub description: String,
}

/// Get all valid state transitions
pub fn get_all_transitions() -> Vec<StateTransition> {
    vec![
        // Queen transitions
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            description: "Queen: Stopped → Running (hive stays stopped)".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            description: "Queen: Running → Stopped".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            description: "Queen: Running → Running (idempotent)".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            description: "Queen: Stopped → Stopped (idempotent)".to_string(),
        },
        
        // Hive transitions
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["hive".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Hive start: Both stopped → Both running".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            command: vec!["hive".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Hive start: Queen running → Hive starts".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            command: vec!["hive".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Hive start: Both running (idempotent)".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            command: vec!["hive".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            description: "Hive stop: Hive stops, queen stays running".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            command: vec!["hive".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            description: "Hive stop: Already stopped (idempotent)".to_string(),
        },
        
        // Cascade transitions
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            command: vec!["queen".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            description: "Queen stop cascades: Both running → Both stopped".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["hive".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            description: "Hive stop when both stopped (idempotent)".to_string(),
        },
        
        // Additional edge cases
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            command: vec!["queen".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Queen start when already running (idempotent)".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            description: "Queen start from clean state".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            command: vec!["hive".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            description: "Hive stop leaves queen running".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            command: vec!["hive".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Hive start when queen already running".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            command: vec!["hive".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Hive start when already running (idempotent)".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            description: "Queen stop when already stopped (idempotent)".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Stopped,
            },
            command: vec!["queen".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            description: "Queen stop when hive already stopped".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            command: vec!["hive".to_string(), "start".to_string()],
            expected: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            description: "Hive start from completely stopped state".to_string(),
        },
        StateTransition {
            initial: SystemState {
                queen: ProcessState::Running,
                hive: ProcessState::Running,
            },
            command: vec!["queen".to_string(), "stop".to_string()],
            expected: SystemState {
                queen: ProcessState::Stopped,
                hive: ProcessState::Stopped,
            },
            description: "Queen stop cascades to hive".to_string(),
        },
    ]
}

#[tokio::test]
async fn test_all_state_transitions() {
    // TEAM-252: Test all state transitions systematically
    
    let transitions = get_all_transitions();
    let total = transitions.len();
    
    println!("🔄 Testing {} state transitions...\n", total);
    
    for (i, transition) in transitions.iter().enumerate() {
        println!("📝 Test {}/{}: {}", i + 1, total, transition.description);
        
        let mut harness = TestHarness::new().await.unwrap();
        
        // Set up initial state
        harness.set_state(transition.initial.clone()).await.unwrap();
        
        // Verify initial state
        let actual_initial = harness.get_state().await;
        assert_eq!(actual_initial, transition.initial, 
            "Failed to set up initial state for: {}", transition.description);
        
        // Execute command
        let cmd_refs: Vec<&str> = transition.command.iter().map(|s| s.as_str()).collect();
        let result = harness.run_command(&cmd_refs).await.unwrap();
        assert_success(&result);
        
        // Wait for state to stabilize
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        // Verify final state
        let actual_final = harness.get_state().await;
        assert_eq!(actual_final, transition.expected, 
            "State transition failed: {}\nExpected: {:?}\nActual: {:?}",
            transition.description, transition.expected, actual_final);
        
        println!("✅ Passed\n");
        
        harness.cleanup().await.unwrap();
    }
    
    println!("🎉 All {} state transitions passed!", total);
}

#[tokio::test]
async fn test_transition_queen_start_from_stopped() {
    // TEAM-252: Test queen start from stopped state
    let mut harness = TestHarness::new().await.unwrap();
    
    // Verify initial state
    let initial = harness.get_state().await;
    assert_eq!(initial.queen, ProcessState::Stopped);
    
    // Start queen
    let result = harness.run_command(&["queen", "start"]).await.unwrap();
    assert_success(&result);
    
    // Wait for startup
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Verify queen is running
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Running);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_queen_stop_from_running() {
    // TEAM-252: Test queen stop from running state
    let mut harness = TestHarness::new().await.unwrap();
    
    // Start queen
    harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    
    // Stop queen
    let result = harness.run_command(&["queen", "stop"]).await.unwrap();
    assert_success(&result);
    
    // Wait for shutdown
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Verify queen is stopped
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Stopped);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_hive_start_both_stopped() {
    // TEAM-252: Test hive start when both are stopped
    let mut harness = TestHarness::new().await.unwrap();
    
    // Verify both stopped
    let initial = harness.get_state().await;
    assert_eq!(initial.queen, ProcessState::Stopped);
    assert_eq!(initial.hive, ProcessState::Stopped);
    
    // Start hive (should start queen too)
    let result = harness.run_command(&["hive", "start"]).await.unwrap();
    assert_success(&result);
    
    // Wait for startup
    tokio::time::sleep(Duration::from_secs(3)).await;
    
    // Verify both running
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Running);
    assert_eq!(final_state.hive, ProcessState::Running);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_hive_stop_leaves_queen() {
    // TEAM-252: Test hive stop leaves queen running
    let mut harness = TestHarness::new().await.unwrap();
    
    // Start both
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    
    // Stop hive
    let result = harness.run_command(&["hive", "stop"]).await.unwrap();
    assert_success(&result);
    
    // Wait for shutdown
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Verify hive stopped but queen running
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Running);
    assert_eq!(final_state.hive, ProcessState::Stopped);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_queen_stop_cascades_to_hive() {
    // TEAM-252: Test queen stop cascades to hive
    let mut harness = TestHarness::new().await.unwrap();
    
    // Start both
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    
    // Verify both running
    let initial = harness.get_state().await;
    assert_eq!(initial.queen, ProcessState::Running);
    assert_eq!(initial.hive, ProcessState::Running);
    
    // Stop queen
    let result = harness.run_command(&["queen", "stop"]).await.unwrap();
    assert_success(&result);
    
    // Wait for cascade
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Verify both stopped
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Stopped);
    assert_eq!(final_state.hive, ProcessState::Stopped);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_idempotent_queen_start() {
    // TEAM-252: Test queen start is idempotent
    let mut harness = TestHarness::new().await.unwrap();
    
    // Start queen
    harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    
    // Start queen again (should be idempotent)
    let result = harness.run_command(&["queen", "start"]).await.unwrap();
    assert_success(&result);
    
    // Verify queen still running
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Running);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_idempotent_queen_stop() {
    // TEAM-252: Test queen stop is idempotent
    let mut harness = TestHarness::new().await.unwrap();
    
    // Stop queen when already stopped (should be idempotent)
    let result = harness.run_command(&["queen", "stop"]).await.unwrap();
    assert_success(&result);
    
    // Verify queen still stopped
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Stopped);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_idempotent_hive_start() {
    // TEAM-252: Test hive start is idempotent
    let mut harness = TestHarness::new().await.unwrap();
    
    // Start hive
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    
    // Start hive again (should be idempotent)
    let result = harness.run_command(&["hive", "start"]).await.unwrap();
    assert_success(&result);
    
    // Verify both still running
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Running);
    assert_eq!(final_state.hive, ProcessState::Running);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_idempotent_hive_stop() {
    // TEAM-252: Test hive stop is idempotent
    let mut harness = TestHarness::new().await.unwrap();
    
    // Stop hive when already stopped (should be idempotent)
    let result = harness.run_command(&["hive", "stop"]).await.unwrap();
    assert_success(&result);
    
    // Verify hive still stopped
    let final_state = harness.get_state().await;
    assert_eq!(final_state.hive, ProcessState::Stopped);
    
    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_transition_matrix_all_combinations() {
    // TEAM-252: Test all possible state combinations
    let states = vec![
        (ProcessState::Stopped, ProcessState::Stopped),
        (ProcessState::Running, ProcessState::Stopped),
        (ProcessState::Running, ProcessState::Running),
    ];
    
    for (queen_state, hive_state) in states {
        let mut harness = TestHarness::new().await.unwrap();
        
        let target_state = SystemState {
            queen: queen_state.clone(),
            hive: hive_state.clone(),
        };
        
        harness.set_state(target_state.clone()).await.unwrap();
        
        let actual = harness.get_state().await;
        assert_eq!(actual, target_state, 
            "Failed to reach state: queen={:?}, hive={:?}", queen_state, hive_state);
        
        harness.cleanup().await.unwrap();
    }
}

#[tokio::test]
async fn test_invalid_transition_hive_without_queen() {
    // TEAM-252: Test that hive start auto-starts queen
    let mut harness = TestHarness::new().await.unwrap();
    
    // Verify both stopped
    let initial = harness.get_state().await;
    assert_eq!(initial.queen, ProcessState::Stopped);
    assert_eq!(initial.hive, ProcessState::Stopped);
    
    // Try to start hive (should auto-start queen)
    let result = harness.run_command(&["hive", "start"]).await.unwrap();
    assert_success(&result);
    
    // Wait for startup
    tokio::time::sleep(Duration::from_secs(3)).await;
    
    // Verify queen was auto-started
    let final_state = harness.get_state().await;
    assert_eq!(final_state.queen, ProcessState::Running, 
        "Queen should be auto-started when hive starts");
    assert_eq!(final_state.hive, ProcessState::Running);
    
    harness.cleanup().await.unwrap();
}
