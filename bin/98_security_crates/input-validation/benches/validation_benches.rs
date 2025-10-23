use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use input_validation::{
    sanitize_string, validate_hex_string, validate_identifier, validate_model_ref, validate_prompt,
    validate_range,
};

fn bench_validate_identifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_identifier");

    // Pre-generate inputs to avoid allocation inside the measured loop
    let sizes = [8usize, 32, 128, 256];
    for &size in &sizes {
        let s = format!("id-{}", "a".repeat(size.saturating_sub(3)));
        group.bench_with_input(BenchmarkId::new("valid", size), &s, |b, s| {
            b.iter(|| {
                let _ = black_box(validate_identifier(black_box(s.as_str()), 256)).unwrap();
            });
        });
    }

    // Invalid cases hit fast-fail paths
    let invalids = [
        ("null_byte", String::from("bad\0id")),
        ("path_traversal", String::from(".. /etc".replace(' ', ""))), // "../etc"
        ("invalid_char", String::from("shard@123")),
        ("too_long", format!("a{}", "a".repeat(300))),
    ];
    for (name, s) in invalids {
        group.bench_with_input(BenchmarkId::new("invalid", name), &s, |b, s| {
            b.iter(|| {
                let _ = black_box(validate_identifier(black_box(s.as_str()), 256)).is_err();
            });
        });
    }

    group.finish();
}

fn bench_validate_model_ref(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_model_ref");

    let valids = [
        "meta-llama/Llama-3.1-8B",
        "hf:org/repo",
        "file:models/model.gguf",
        "https://example.com/model.bin",
        "model-v1.2.3",
    ];
    for s in &valids {
        group.bench_with_input(BenchmarkId::new("valid", *s), s, |b, s| {
            b.iter(|| {
                let _ = black_box(validate_model_ref(black_box(s))).unwrap();
            });
        });
    }

    let invalids = [
        ("shell_semicolon", "model;rm"),
        ("newline", "model\nlog"),
        ("space", "model name"),
        ("path_traversal", "../../../etc/passwd"),
        ("backslash_traversal", "..\\windows"),
        ("unicode", "modèl"),
    ];
    for (name, s) in &invalids {
        group.bench_with_input(BenchmarkId::new("invalid", *name), s, |b, s| {
            b.iter(|| {
                let _ = black_box(validate_model_ref(black_box(s))).is_err();
            });
        });
    }

    group.finish();
}

fn bench_validate_hex_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_hex_string");

    // Common digest lengths
    let lengths = [32usize, 40, 64, 256];
    for &len in &lengths {
        let s = "a".repeat(len);
        group.bench_with_input(BenchmarkId::new("valid", len), &s, |b, s| {
            b.iter(|| {
                let _ = black_box(validate_hex_string(black_box(s.as_str()), len)).unwrap();
            });
        });
    }

    // Invalid: wrong length and invalid chars
    let invalids = [
        ("wrong_len", ("a".repeat(10), 64usize)),
        ("invalid_char", (String::from("abcg"), 4usize)),
        ("space", (String::from("ab cd"), 5usize)),
        ("null_byte", (String::from("ab\0cd"), 5usize)),
    ];
    for (name, (s, expected)) in &invalids {
        group.bench_with_input(BenchmarkId::new("invalid", *name), s, |b, s| {
            b.iter(|| {
                let _ = black_box(validate_hex_string(black_box(s.as_str()), *expected)).is_err();
            });
        });
    }

    group.finish();
}

fn bench_sanitize_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("sanitize_string");

    let valids = [
        "normal text",
        "text with\ttab",
        "line1\nline2\r\n",
        "café ☕", // still valid; sanitize allows unicode other than directional overrides and BOM
    ];
    for s in &valids {
        group.bench_with_input(BenchmarkId::new("valid", *s), s, |b, s| {
            b.iter(|| {
                let _ = black_box(sanitize_string(black_box(s))).unwrap();
            });
        });
    }

    let invalids = [
        ("null_byte", "text\0null"),
        ("ansi", "text\x1b[31mred"),
        ("control", "text\x07bell"),
        ("unicode_override", "text\u{202E}rev"),
        ("bom", "\u{FEFF}text"),
    ];
    for (name, s) in &invalids {
        group.bench_with_input(BenchmarkId::new("invalid", *name), s, |b, s| {
            b.iter(|| {
                let _ = black_box(sanitize_string(black_box(s))).is_err();
            });
        });
    }

    group.finish();
}

fn bench_validate_prompt(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_prompt");

    // Typical prompt sizes
    let sizes = [128usize, 1024, 8192, 32_768];
    for &size in &sizes {
        let s = "a".repeat(size);
        group.bench_with_input(BenchmarkId::new("valid", size), &s, |b, s| {
            b.iter(|| {
                let _ = black_box(validate_prompt(black_box(s.as_str()), 100_000)).unwrap();
            });
        });
    }

    // Invalid prompts
    let over = "a".repeat(120_000);
    let with_null = String::from("prompt\0null");
    let with_ansi = String::from("text\x1b[31mred");

    group.bench_with_input(BenchmarkId::new("invalid", "too_long"), &over, |b, s| {
        b.iter(|| {
            let _ = black_box(validate_prompt(black_box(s.as_str()), 100_000)).is_err();
        });
    });
    group.bench_with_input(BenchmarkId::new("invalid", "null_byte"), &with_null, |b, s| {
        b.iter(|| {
            let _ = black_box(validate_prompt(black_box(s.as_str()), 100_000)).is_err();
        });
    });
    group.bench_with_input(BenchmarkId::new("invalid", "ansi"), &with_ansi, |b, s| {
        b.iter(|| {
            let _ = black_box(validate_prompt(black_box(s.as_str()), 100_000)).is_err();
        });
    });

    group.finish();
}

fn bench_validate_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("validate_range");

    // Integers
    group.bench_function("int_in_range", |b| {
        b.iter(|| {
            let _ = black_box(validate_range(black_box(1024), 1, 4096)).unwrap();
        });
    });
    group.bench_function("int_out_of_range", |b| {
        b.iter(|| {
            let _ = black_box(validate_range(black_box(5000), 1, 4096)).is_err();
        });
    });

    // Floats
    group.bench_function("float_in_range", |b| {
        b.iter(|| {
            let _ = black_box(validate_range(black_box(0.5f64), 0.0, 1.0)).unwrap();
        });
    });
    group.bench_function("float_out_of_range", |b| {
        b.iter(|| {
            let _ = black_box(validate_range(black_box(1.1f64), 0.0, 1.0)).is_err();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_validate_identifier,
    bench_validate_model_ref,
    bench_validate_hex_string,
    bench_sanitize_string,
    bench_validate_prompt,
    bench_validate_range
);
criterion_main!(benches);
