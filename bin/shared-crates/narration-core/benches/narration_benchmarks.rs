use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use observability_narration_core::{
    narrate, NarrationFields, NarrationLevel, narrate_at_level,
    redact_secrets, RedactionPolicy,
    generate_correlation_id, validate_correlation_id,
    sanitize_crlf, sanitize_for_json,
};

fn bench_template_interpolation(c: &mut Criterion) {
    c.bench_function("template_interpolation_simple", |b| {
        b.iter(|| {
            let job_id = black_box("job-123");
            let worker_id = black_box("worker-gpu0-r1");
            format!("Dispatched job {} to worker {}", job_id, worker_id)
        });
    });
}

fn bench_redaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("redaction");
    
    // Clean string (no secrets)
    let clean_text = "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'";
    group.bench_function("clean_1000_chars", |b| {
        let text = clean_text.repeat(10); // ~1000 chars
        b.iter(|| {
            redact_secrets(black_box(&text), RedactionPolicy::default())
        });
    });
    
    // With bearer token
    let bearer_text = "Authorization: Bearer abc123xyz";
    group.bench_function("with_bearer_token", |b| {
        b.iter(|| {
            redact_secrets(black_box(bearer_text), RedactionPolicy::default())
        });
    });
    
    // With multiple secrets
    let multi_secrets = "Bearer token123 and api_key=secret456 and jwt eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
    group.bench_function("with_multiple_secrets", |b| {
        b.iter(|| {
            redact_secrets(black_box(multi_secrets), RedactionPolicy::default())
        });
    });
    
    group.finish();
}

fn bench_crlf_sanitization(c: &mut Criterion) {
    let mut group = c.benchmark_group("crlf_sanitization");
    
    // Clean string (no CRLF)
    let clean = "No newlines here at all";
    group.bench_function("clean_string", |b| {
        b.iter(|| {
            sanitize_crlf(black_box(clean))
        });
    });
    
    // With newlines
    let with_newlines = "Line 1\nLine 2\rLine 3\tTab";
    group.bench_function("with_newlines", |b| {
        b.iter(|| {
            sanitize_crlf(black_box(with_newlines))
        });
    });
    
    group.finish();
}

fn bench_unicode_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("unicode_validation");
    
    // ASCII fast path
    let ascii_text = "Hello, world! This is a test message.";
    group.bench_function("ascii_fast_path", |b| {
        b.iter(|| {
            sanitize_for_json(black_box(ascii_text))
        });
    });
    
    // With emoji
    let emoji_text = "Hello 🎀 world! This is cute! ✨";
    group.bench_function("with_emoji", |b| {
        b.iter(|| {
            sanitize_for_json(black_box(emoji_text))
        });
    });
    
    group.finish();
}

fn bench_correlation_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_id");
    
    // Generation
    group.bench_function("generate", |b| {
        b.iter(|| {
            generate_correlation_id()
        });
    });
    
    // Validation (valid UUID)
    let valid_uuid = "550e8400-e29b-41d4-a716-446655440000";
    group.bench_function("validate_valid", |b| {
        b.iter(|| {
            validate_correlation_id(black_box(valid_uuid))
        });
    });
    
    // Validation (invalid)
    let invalid = "not-a-uuid";
    group.bench_function("validate_invalid", |b| {
        b.iter(|| {
            validate_correlation_id(black_box(invalid))
        });
    });
    
    group.finish();
}

fn bench_narration_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("narration_levels");
    
    for level in &[
        ("info", NarrationLevel::Info),
        ("warn", NarrationLevel::Warn),
        ("error", NarrationLevel::Error),
    ] {
        group.bench_with_input(BenchmarkId::from_parameter(level.0), &level.1, |b, &lvl| {
            b.iter(|| {
                narrate_at_level(NarrationFields {
                    actor: "benchmark",
                    action: "test",
                    target: "target".to_string(),
                    human: "Benchmark test message".to_string(),
                    ..Default::default()
                }, lvl)
            });
        });
    }
    
    group.finish();
}

fn bench_trace_macros(c: &mut Criterion) {
    #[cfg(feature = "trace-enabled")]
    {
        use observability_narration_core::{trace_tiny, trace_enter, trace_exit};
        
        c.bench_function("trace_tiny", |b| {
            b.iter(|| {
                trace_tiny!("benchmark", "test", "target", "Test message");
            });
        });
        
        c.bench_function("trace_enter", |b| {
            b.iter(|| {
                trace_enter!("benchmark", "test_function", "arg1=value1");
            });
        });
        
        c.bench_function("trace_exit", |b| {
            b.iter(|| {
                trace_exit!("benchmark", "test_function", "→ Ok (5ms)");
            });
        });
    }
    
    #[cfg(not(feature = "trace-enabled"))]
    {
        c.bench_function("trace_disabled_overhead", |b| {
            b.iter(|| {
                // Measure overhead of disabled trace macros (should be 0ns)
                black_box(());
            });
        });
    }
}

criterion_group!(
    benches,
    bench_template_interpolation,
    bench_redaction,
    bench_crlf_sanitization,
    bench_unicode_validation,
    bench_correlation_id,
    bench_narration_levels,
    bench_trace_macros,
);

criterion_main!(benches);
