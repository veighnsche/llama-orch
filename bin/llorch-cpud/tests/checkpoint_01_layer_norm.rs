//! Checkpoint 1: LayerNorm Test
//!
//! Tests LayerNorm implementation against expected behavior

use llorch_cpud::layers::LayerNorm;
use ndarray::{Array, Array1, Array2, Axis};

#[test]
fn test_layer_norm_shape() {
    let dim = 1024;
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);

    let input = Array2::zeros((2, dim));
    let output = ln.forward(&input);

    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_layer_norm_mean_variance() {
    let dim = 4;
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);

    // Create input with known mean and variance
    let input = Array::from_shape_vec((1, dim), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = ln.forward(&input);

    // After layer norm, mean should be ~0 and variance should be ~1
    let output_mean = output.mean_axis(Axis(1)).unwrap()[[0]];
    let output_centered = &output - output_mean;
    let output_var = (&output_centered * &output_centered).mean_axis(Axis(1)).unwrap()[[0]];

    assert!((output_mean.abs()) < 1e-5, "Mean should be ~0, got {}", output_mean);
    assert!((output_var - 1.0).abs() < 1e-4, "Variance should be ~1, got {}", output_var);
}

#[test]
fn test_layer_norm_with_scale_bias() {
    let dim = 4;
    let weight = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0]);
    let bias = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let ln = LayerNorm::new(weight, bias, 1e-5);

    let input = Array::from_shape_vec((1, dim), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = ln.forward(&input);

    // With scale=2 and bias=1, output should be scaled and shifted
    assert_eq!(output.shape(), &[1, dim]);

    // Mean should be bias = 1.0 (after scaling and shifting)
    let output_mean = output.mean_axis(Axis(1)).unwrap()[[0]];
    assert!((output_mean - 1.0).abs() < 1e-4, "Mean should be ~1.0, got {}", output_mean);
}

#[test]
fn test_layer_norm_batch() {
    let dim = 4;
    let batch_size = 3;
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);

    let input = Array2::from_shape_fn((batch_size, dim), |(i, j)| (i * dim + j) as f32);
    let output = ln.forward(&input);

    assert_eq!(output.shape(), &[batch_size, dim]);

    // Each row should be normalized independently
    for i in 0..batch_size {
        let row = output.row(i);
        let row_mean = row.mean().unwrap();
        let row_var = row.mapv(|x| (x - row_mean).powi(2)).mean().unwrap();

        assert!((row_mean.abs()) < 1e-5, "Row {} mean should be ~0, got {}", i, row_mean);
        assert!((row_var - 1.0).abs() < 1e-4, "Row {} variance should be ~1, got {}", i, row_var);
    }
}
