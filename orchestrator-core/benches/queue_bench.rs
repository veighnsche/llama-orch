use criterion::{black_box, criterion_group, criterion_main, Criterion};
use orchestrator_core::queue::{InMemoryQueue, Policy, Priority};

fn bench_enqueue_interactive(c: &mut Criterion) {
    c.bench_function("enqueue_interactive_10k", |b| {
        b.iter(|| {
            let mut q = InMemoryQueue::with_capacity_policy(10_240, Policy::Reject);
            for i in 0..10_000u32 {
                let _ = q.enqueue(black_box(i), Priority::Interactive);
            }
            black_box(q.len())
        })
    });
}

fn bench_enqueue_drop_lru(c: &mut Criterion) {
    c.bench_function("enqueue_drop_lru_full_4k", |b| {
        b.iter(|| {
            let mut q = InMemoryQueue::with_capacity_policy(4_096, Policy::DropLru);
            for i in 0..6_000u32 {
                let _ = q.enqueue(black_box(i), Priority::Batch);
            }
            black_box(q.len())
        })
    });
}

criterion_group!(benches, bench_enqueue_interactive, bench_enqueue_drop_lru);
criterion_main!(benches);
