//! Integration tests for GPU detection

use gpu_info::detect_gpus;

#[test]
fn test_detect_gpus_does_not_panic() {
    // Should not panic regardless of GPU availability
    let info = detect_gpus();
    
    if info.available {
        println!("✅ Detected {} GPU(s)", info.count);
        for gpu in &info.devices {
            println!(
                "   GPU {}: {} ({:.1} GB VRAM, {:.1} GB free)",
                gpu.index,
                gpu.name,
                gpu.vram_total_gb(),
                gpu.vram_free_gb()
            );
        }
    } else {
        println!("💻 No GPU detected (expected in CI)");
    }
}

#[test]
fn test_gpu_info_consistency() {
    let info = detect_gpus();
    
    // Consistency checks
    assert_eq!(info.available, !info.devices.is_empty());
    assert_eq!(info.count, info.devices.len());
    
    // Device indices should be sequential
    for (i, device) in info.devices.iter().enumerate() {
        assert_eq!(device.index as usize, i);
    }
}

#[test]
fn test_vram_calculations() {
    let info = detect_gpus();
    
    if !info.available {
        return; // Skip if no GPU
    }
    
    // Total VRAM should equal sum of device VRAM
    let sum: usize = info.devices.iter().map(|d| d.vram_total_bytes).sum();
    assert_eq!(info.total_vram_bytes(), sum);
    
    // Free VRAM should equal sum of device free VRAM
    let free_sum: usize = info.devices.iter().map(|d| d.vram_free_bytes).sum();
    assert_eq!(info.total_free_vram_bytes(), free_sum);
    
    // Free VRAM should be less than or equal to total
    for device in &info.devices {
        assert!(device.vram_free_bytes <= device.vram_total_bytes);
    }
}

#[test]
fn test_best_gpu_selection() {
    let info = detect_gpus();
    
    if info.count < 2 {
        return; // Skip if less than 2 GPUs
    }
    
    let best = info.best_gpu_for_workload().unwrap();
    
    // Best GPU should have most free VRAM
    for device in &info.devices {
        assert!(best.vram_free_bytes >= device.vram_free_bytes);
    }
}

#[test]
fn test_device_validation() {
    let info = detect_gpus();
    
    if !info.available {
        // Should fail if no GPU
        assert!(info.validate_device(0).is_err());
        return;
    }
    
    // Valid devices
    for i in 0..info.count {
        assert!(info.validate_device(i as u32).is_ok());
    }
    
    // Invalid devices
    assert!(info.validate_device(info.count as u32).is_err());
    assert!(info.validate_device(999).is_err());
}
