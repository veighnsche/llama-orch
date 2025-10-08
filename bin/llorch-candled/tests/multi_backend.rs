//! Multi-backend integration tests
//!
//! Tests device initialization for each backend.
//! Each test is feature-gated to its respective backend.
//!
//! Created by: TEAM-007

#[cfg(feature = "cpu")]
#[test]
fn test_cpu_device_init() {
    use llorch_candled::device::{init_cpu_device, verify_device};
    
    let device = init_cpu_device().expect("Failed to initialize CPU device");
    verify_device(&device).expect("Failed to verify CPU device");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_device_init() {
    use llorch_candled::device::{init_cuda_device, verify_device};
    
    // Only run if CUDA is available
    if let Ok(device) = init_cuda_device(0) {
        verify_device(&device).expect("Failed to verify CUDA device");
    } else {
        println!("CUDA not available, skipping test");
    }
}

#[cfg(feature = "accelerate")]
#[test]
fn test_accelerate_device_init() {
    use llorch_candled::device::{init_accelerate_device, verify_device};
    
    let device = init_accelerate_device().expect("Failed to initialize Accelerate device");
    verify_device(&device).expect("Failed to verify Accelerate device");
}

#[cfg(feature = "cpu")]
#[test]
fn test_cpu_tensor_operations() {
    use llorch_candled::device::init_cpu_device;
    use candle_core::Tensor;
    
    let device = init_cpu_device().expect("Failed to initialize CPU device");
    
    // Test basic tensor operations
    let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).expect("Failed to create tensor");
    let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device).expect("Failed to create tensor");
    
    let sum = (&a + &b).expect("Failed to add tensors");
    let expected = vec![5.0f32, 7.0, 9.0];
    let result: Vec<f32> = sum.to_vec1().expect("Failed to convert to vec");
    
    assert_eq!(result, expected, "Tensor addition failed on CPU");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_tensor_operations() {
    use llorch_candled::device::init_cuda_device;
    use candle_core::Tensor;
    
    // Only run if CUDA is available
    if let Ok(device) = init_cuda_device(0) {
        // Test basic tensor operations
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).expect("Failed to create tensor");
        let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device).expect("Failed to create tensor");
        
        let sum = (&a + &b).expect("Failed to add tensors");
        let expected = vec![5.0f32, 7.0, 9.0];
        let result: Vec<f32> = sum.to_vec1().expect("Failed to convert to vec");
        
        assert_eq!(result, expected, "Tensor addition failed on CUDA");
    } else {
        println!("CUDA not available, skipping test");
    }
}

#[cfg(feature = "accelerate")]
#[test]
fn test_accelerate_tensor_operations() {
    use llorch_candled::device::init_accelerate_device;
    use candle_core::Tensor;
    
    let device = init_accelerate_device().expect("Failed to initialize Accelerate device");
    
    // Test basic tensor operations
    let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device).expect("Failed to create tensor");
    let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device).expect("Failed to create tensor");
    
    let sum = (&a + &b).expect("Failed to add tensors");
    let expected = vec![5.0f32, 7.0, 9.0];
    let result: Vec<f32> = sum.to_vec1().expect("Failed to convert to vec");
    
    assert_eq!(result, expected, "Tensor addition failed on Accelerate");
}
