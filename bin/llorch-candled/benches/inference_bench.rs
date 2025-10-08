// Created by: TEAM-006
// Benchmark suite for llorch-candled inference components

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use llorch_candled::layers::{RoPE, QKVProjection, Attention};
use candle_core::{Tensor, Device};

fn bench_rope(c: &mut Criterion) {
    let device = Device::Cpu;
    let rope = RoPE::new(128, 4096, 10000.0, &device).unwrap();
    
    // Test different sequence lengths
    for seq_len in [1, 8, 32, 128].iter() {
        let q = Tensor::randn(0f32, 1.0, (1, *seq_len, 32, 128), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, *seq_len, 32, 128), &device).unwrap();
        
        c.bench_with_input(
            BenchmarkId::new("rope_forward", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    rope.forward(black_box(&q), black_box(&k), 0).unwrap()
                })
            },
        );
    }
}

fn bench_qkv_projection(c: &mut Criterion) {
    let device = Device::Cpu;
    let hidden_size = 4096;
    let n_heads = 32;
    
    let q_weight = vec![0.1f32; hidden_size * hidden_size];
    let k_weight = vec![0.1f32; hidden_size * hidden_size];
    let v_weight = vec![0.1f32; hidden_size * hidden_size];
    
    let qkv = QKVProjection::from_arrays(
        &q_weight,
        &k_weight,
        &v_weight,
        hidden_size,
        n_heads,
        &device,
    ).unwrap();
    
    for seq_len in [1, 8, 32, 128].iter() {
        let input = Tensor::randn(0f32, 1.0, (1, *seq_len, hidden_size), &device).unwrap();
        
        c.bench_with_input(
            BenchmarkId::new("qkv_projection", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    qkv.forward(black_box(&input)).unwrap()
                })
            },
        );
    }
}

fn bench_attention_scores(c: &mut Criterion) {
    let device = Device::Cpu;
    let hidden_size = 4096;
    let n_heads = 32;
    
    let q_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    let k_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    let v_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    
    let attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device).unwrap();
    
    for seq_len in [1, 8, 32, 128].iter() {
        let q = Tensor::randn(0f32, 1.0, (1, *seq_len, n_heads, 128), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, *seq_len, n_heads, 128), &device).unwrap();
        
        c.bench_with_input(
            BenchmarkId::new("attention_scores", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    attn.compute_scores(black_box(&q), black_box(&k)).unwrap()
                })
            },
        );
    }
}

fn bench_causal_mask(c: &mut Criterion) {
    let device = Device::Cpu;
    let hidden_size = 4096;
    let n_heads = 32;
    
    let q_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    let k_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    let v_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    
    let mut attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device).unwrap();
    
    for seq_len in [8, 32, 128, 512].iter() {
        let scores = Tensor::randn(0f32, 1.0, (1, n_heads, *seq_len, *seq_len), &device).unwrap();
        
        c.bench_with_input(
            BenchmarkId::new("causal_mask", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    attn.apply_causal_mask(black_box(&scores)).unwrap()
                })
            },
        );
    }
}

fn bench_full_attention(c: &mut Criterion) {
    let device = Device::Cpu;
    let hidden_size = 4096;
    let n_heads = 32;
    
    let q_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    let k_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    let v_weight = Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device).unwrap();
    
    let mut attn = Attention::new(q_weight, k_weight, v_weight, n_heads, &device).unwrap();
    
    for seq_len in [1, 8, 32].iter() {
        let q = Tensor::randn(0f32, 1.0, (1, *seq_len, n_heads, 128), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, *seq_len, n_heads, 128), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, *seq_len, n_heads, 128), &device).unwrap();
        
        c.bench_with_input(
            BenchmarkId::new("full_attention", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    attn.forward(black_box(&q), black_box(&k), black_box(&v), true).unwrap()
                })
            },
        );
    }
}

criterion_group!(
    benches,
    bench_rope,
    bench_qkv_projection,
    bench_attention_scores,
    bench_causal_mask,
    bench_full_attention
);
criterion_main!(benches);
