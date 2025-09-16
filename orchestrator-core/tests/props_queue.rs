//! Property-based test scaffolds for queue invariants.
//! Stage 3: These property tests exercise basic queue invariants in `orchestrator-core`.

use orchestrator_core::queue::{EnqueueError, InMemoryQueue, Policy, Priority};
use proptest::prelude::*;
use std::collections::HashSet;

fn arb_priority() -> impl Strategy<Value = Priority> {
    prop_oneof![Just(Priority::Interactive), Just(Priority::Batch)]
}

fn arb_enqueue_pairs() -> impl Strategy<Value = Vec<(u32, Priority)>> {
    prop::collection::vec((0u32..10_000u32, arb_priority()), 1..128)
}

proptest! {
    #[test]
    fn fifo_within_same_priority_enqueue_only(pairs in arb_enqueue_pairs()) {
        // Capacity is large enough to avoid full behavior.
        let mut q = InMemoryQueue::with_capacity_policy(pairs.len().saturating_add(8), Policy::Reject);
        for (id, prio) in &pairs { let _ = q.enqueue(*id, *prio); }

        let expected_interactive: Vec<u32> = pairs.iter().filter_map(|(id, p)| if *p == Priority::Interactive { Some(*id) } else { None }).collect();
        let expected_batch: Vec<u32> = pairs.iter().filter_map(|(id, p)| if *p == Priority::Batch { Some(*id) } else { None }).collect();

        prop_assert_eq!(q.snapshot_priority(Priority::Interactive), expected_interactive);
        prop_assert_eq!(q.snapshot_priority(Priority::Batch), expected_batch);
    }
}

proptest! {
    #[test]
    #[ignore]
    fn priority_fairness_placeholder(_pairs in arb_enqueue_pairs()) {
        // Placeholder: budget or fair-share for interactive over batch under contention.
        prop_assert!(true);
    }
}

proptest! {
    #[test]
    fn reject_vs_drop_lru_semantics(
        b1 in 1u32..10000,
        b2 in 1u32..10000,
        i1 in 1u32..10000,
        i2 in 1u32..10000,
        i3 in 1u32..10000,
        e in 1u32..10000,
        e_is_batch in any::<bool>(),
    ) {
        // Ensure distinct ids to make ordering assertions meaningful.
        let ids: HashSet<u32> = [b1,b2,i1,i2,i3,e].into_iter().collect();
        prop_assume!(ids.len() == 6);

        // Build initial full queue state with capacity 5
        let mut q_drop = InMemoryQueue::with_capacity_policy(5, Policy::DropLru);
        let _ = q_drop.enqueue(b1, Priority::Batch);
        let _ = q_drop.enqueue(b2, Priority::Batch);
        let _ = q_drop.enqueue(i1, Priority::Interactive);
        let _ = q_drop.enqueue(i2, Priority::Interactive);
        let _ = q_drop.enqueue(i3, Priority::Interactive);
        prop_assert_eq!(q_drop.snapshot_priority(Priority::Batch), vec![b1,b2]);
        prop_assert_eq!(q_drop.snapshot_priority(Priority::Interactive), vec![i1,i2,i3]);

        // On enqueue when full with DropLru: drop oldest batch if present; otherwise drop oldest overall.
        let prio_e = if e_is_batch { Priority::Batch } else { Priority::Interactive };
        let _ = q_drop.enqueue(e, prio_e);

        // Expected after drop-lru: b1 removed; e appended to its priority queue
        let mut exp_batch = vec![b2];
        let mut exp_inter = vec![i1,i2,i3];
        match prio_e {
            Priority::Batch => exp_batch.push(e),
            Priority::Interactive => exp_inter.push(e),
        }
        prop_assert_eq!(q_drop.snapshot_priority(Priority::Batch), exp_batch);
        prop_assert_eq!(q_drop.snapshot_priority(Priority::Interactive), exp_inter);

        // For Reject policy: enqueue returns error and state unchanged
        let mut q_reject = InMemoryQueue::with_capacity_policy(5, Policy::Reject);
        let _ = q_reject.enqueue(b1, Priority::Batch);
        let _ = q_reject.enqueue(b2, Priority::Batch);
        let _ = q_reject.enqueue(i1, Priority::Interactive);
        let _ = q_reject.enqueue(i2, Priority::Interactive);
        let _ = q_reject.enqueue(i3, Priority::Interactive);
        let before_batch = q_reject.snapshot_priority(Priority::Batch);
        let before_inter = q_reject.snapshot_priority(Priority::Interactive);
        let res = q_reject.enqueue(e, prio_e);
        prop_assert_eq!(res, Err(EnqueueError::QueueFullReject));
        prop_assert_eq!(q_reject.snapshot_priority(Priority::Batch), before_batch);
        prop_assert_eq!(q_reject.snapshot_priority(Priority::Interactive), before_inter);
    }
}
