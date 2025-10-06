//! Performance Baseline Benchmarks
//!
//! Benchmarks for establishing performance baselines for:
//! - Model loading
//! - Token generation (prefill + decode)
//! - VRAM usage
//! - Throughput (tokens/second)
//!
//! Run with: cargo bench --bench performance_baseline
//!
//! Spec: FT-031

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use worker_orcd::models::{
    gpt::{GPTConfig, GPTWeightLoader},
    phi3::{Phi3Config, Phi3WeightLoader},
    qwen::{QwenConfig, QwenWeightLoader},
    AdapterForwardConfig, LlamaModelAdapter,
};

/// Benchmark model loading time
fn bench_model_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_loading");

    group.bench_function("qwen_0_5b", |b| {
        b.iter(|| {
            let config = QwenConfig::qwen2_5_0_5b();
            let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
            black_box(model)
        });
    });

    group.bench_function("phi3_mini", |b| {
        b.iter(|| {
            let config = Phi3Config::phi3_mini_4k();
            let model = Phi3WeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
            black_box(model)
        });
    });

    group.bench_function("gpt2_small", |b| {
        b.iter(|| {
            let config = GPTConfig::gpt2_small();
            let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
            black_box(model)
        });
    });

    group.finish();
}

/// Benchmark prefill performance
fn bench_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill");

    // Setup models
    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let qwen_model = QwenWeightLoader::load_to_vram("dummy.gguf", &qwen_config).unwrap();
    let qwen_adapter = LlamaModelAdapter::new_qwen(qwen_model);

    let phi3_config = Phi3Config::phi3_mini_4k();
    let phi3_model = Phi3WeightLoader::load_to_vram("dummy.gguf", &phi3_config).unwrap();
    let phi3_adapter = LlamaModelAdapter::new_phi3(phi3_model);

    // Test different sequence lengths
    for seq_len in [32, 128, 512, 1024] {
        let input_ids: Vec<u32> = (0..seq_len).map(|i| i % 1000).collect();
        let config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len,
            cache_len: 0,
            temperature: 1.0,
            seed: 42,
        };

        group.bench_with_input(
            BenchmarkId::new("qwen", seq_len),
            &(input_ids.clone(), config.clone()),
            |b, (ids, cfg)| {
                b.iter(|| black_box(qwen_adapter.prefill(ids, cfg).unwrap()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("phi3", seq_len),
            &(input_ids.clone(), config.clone()),
            |b, (ids, cfg)| {
                b.iter(|| black_box(phi3_adapter.prefill(ids, cfg).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark decode performance
fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let qwen_model = QwenWeightLoader::load_to_vram("dummy.gguf", &qwen_config).unwrap();
    let qwen_adapter = LlamaModelAdapter::new_qwen(qwen_model);

    let phi3_config = Phi3Config::phi3_mini_4k();
    let phi3_model = Phi3WeightLoader::load_to_vram("dummy.gguf", &phi3_config).unwrap();
    let phi3_adapter = LlamaModelAdapter::new_phi3(phi3_model);

    let config = AdapterForwardConfig {
        is_prefill: false,
        batch_size: 1,
        seq_len: 1,
        cache_len: 100,
        temperature: 1.0,
        seed: 42,
    };

    group.bench_function("qwen_decode", |b| {
        b.iter(|| black_box(qwen_adapter.decode(42, &config).unwrap()));
    });

    group.bench_function("phi3_decode", |b| {
        b.iter(|| black_box(phi3_adapter.decode(42, &config).unwrap()));
    });

    group.finish();
}

/// Benchmark generation throughput
fn bench_generation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation_throughput");
    group.sample_size(10); // Reduce sample size for longer benchmarks

    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let qwen_model = QwenWeightLoader::load_to_vram("dummy.gguf", &qwen_config).unwrap();
    let qwen_adapter = LlamaModelAdapter::new_qwen(qwen_model);

    let input_ids = vec![1, 2, 3];

    for max_tokens in [10, 50, 100] {
        let config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: input_ids.len(),
            cache_len: 0,
            temperature: 1.0,
            seed: 42,
        };

        group.bench_with_input(
            BenchmarkId::new("qwen", max_tokens),
            &(input_ids.clone(), max_tokens, config.clone()),
            |b, (ids, tokens, cfg)| {
                b.iter(|| black_box(qwen_adapter.generate(ids, *tokens, cfg).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark VRAM usage queries
fn bench_vram_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("vram_queries");

    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let qwen_model = QwenWeightLoader::load_to_vram("dummy.gguf", &qwen_config).unwrap();
    let qwen_adapter = LlamaModelAdapter::new_qwen(qwen_model);

    group.bench_function("vram_usage", |b| {
        b.iter(|| black_box(qwen_adapter.vram_usage().unwrap()));
    });

    group.bench_function("vocab_size", |b| {
        b.iter(|| black_box(qwen_adapter.vocab_size().unwrap()));
    });

    group.bench_function("hidden_dim", |b| {
        b.iter(|| black_box(qwen_adapter.hidden_dim().unwrap()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_model_loading,
    bench_prefill,
    bench_decode,
    bench_generation_throughput,
    bench_vram_queries
);
criterion_main!(benches);

// ---
// Built by Foundation-Alpha 🏗️
