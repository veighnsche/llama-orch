//! Trait implementations for rbee-heartbeat
//!
//! Created by: TEAM-159
//!
//! Implements the HiveCatalog trait from rbee-heartbeat for the local
//! HiveCatalog implementation.

use crate::{HiveCatalog, HiveRecord, HiveStatus};
use async_trait::async_trait;
use rbee_heartbeat::traits::{
    CatalogError, CpuDevice as TraitCpuDevice, DeviceBackend as TraitDeviceBackend,
    DeviceCapabilities as TraitDeviceCapabilities, GpuDevice as TraitGpuDevice,
    HiveCatalog as HiveCatalogTrait, HiveRecord as TraitHiveRecord, HiveStatus as TraitHiveStatus,
};

// ============================================================================
// Type Conversions
// ============================================================================

/// Convert local HiveStatus to trait HiveStatus
fn convert_status_to_trait(status: HiveStatus) -> TraitHiveStatus {
    match status {
        HiveStatus::Unknown => TraitHiveStatus::Unknown,
        HiveStatus::Online => TraitHiveStatus::Online,
        HiveStatus::Offline => TraitHiveStatus::Offline,
    }
}

/// Convert trait HiveStatus to local HiveStatus
fn convert_status_from_trait(status: TraitHiveStatus) -> HiveStatus {
    match status {
        TraitHiveStatus::Unknown => HiveStatus::Unknown,
        TraitHiveStatus::Online => HiveStatus::Online,
        TraitHiveStatus::Offline => HiveStatus::Offline,
    }
}

/// Convert local HiveRecord to trait HiveRecord
fn convert_hive_to_trait(hive: HiveRecord) -> TraitHiveRecord {
    TraitHiveRecord {
        id: hive.id,
        host: hive.host,
        port: hive.port,
        status: convert_status_to_trait(hive.status),
        last_heartbeat_ms: hive.last_heartbeat_ms,
        devices: hive.devices.map(|d| TraitDeviceCapabilities {
            cpu: d.cpu.map(|cpu| TraitCpuDevice {
                cores: cpu.cores,
                ram_gb: cpu.ram_gb,
            }),
            gpus: d
                .gpus
                .into_iter()
                .map(|gpu| TraitGpuDevice {
                    index: gpu.index,
                    name: gpu.name,
                    vram_gb: gpu.vram_gb,
                    backend: match gpu.backend {
                        crate::DeviceBackend::Cuda => TraitDeviceBackend::Cuda,
                        crate::DeviceBackend::Metal => TraitDeviceBackend::Metal,
                        crate::DeviceBackend::Cpu => TraitDeviceBackend::Cpu,
                    },
                })
                .collect(),
        }),
    }
}

/// Convert trait DeviceCapabilities to local DeviceCapabilities
fn convert_devices_from_trait(devices: TraitDeviceCapabilities) -> crate::DeviceCapabilities {
    crate::DeviceCapabilities {
        cpu: devices.cpu.map(|cpu| crate::CpuDevice {
            cores: cpu.cores,
            ram_gb: cpu.ram_gb,
        }),
        gpus: devices
            .gpus
            .into_iter()
            .map(|gpu| crate::GpuDevice {
                index: gpu.index,
                name: gpu.name,
                vram_gb: gpu.vram_gb,
                backend: match gpu.backend {
                    TraitDeviceBackend::Cuda => crate::DeviceBackend::Cuda,
                    TraitDeviceBackend::Metal => crate::DeviceBackend::Metal,
                    TraitDeviceBackend::Cpu => crate::DeviceBackend::Cpu,
                },
            })
            .collect(),
    }
}

// ============================================================================
// Trait Implementation
// ============================================================================

#[async_trait]
impl HiveCatalogTrait for HiveCatalog {
    async fn update_heartbeat(&self, hive_id: &str, timestamp_ms: i64) -> Result<(), CatalogError> {
        self.update_heartbeat(hive_id, timestamp_ms)
            .await
            .map_err(|e| CatalogError::Database(e.to_string()))
    }

    async fn get_hive(&self, hive_id: &str) -> Result<Option<TraitHiveRecord>, CatalogError> {
        self.get_hive(hive_id)
            .await
            .map(|opt| opt.map(convert_hive_to_trait))
            .map_err(|e| CatalogError::Database(e.to_string()))
    }

    async fn update_devices(
        &self,
        hive_id: &str,
        devices: TraitDeviceCapabilities,
    ) -> Result<(), CatalogError> {
        let local_devices = convert_devices_from_trait(devices);
        self.update_devices(hive_id, local_devices)
            .await
            .map_err(|e| CatalogError::Database(e.to_string()))
    }

    async fn update_hive_status(
        &self,
        hive_id: &str,
        status: TraitHiveStatus,
    ) -> Result<(), CatalogError> {
        let local_status = convert_status_from_trait(status);
        self.update_hive_status(hive_id, local_status)
            .await
            .map_err(|e| CatalogError::Database(e.to_string()))
    }
}
