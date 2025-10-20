// TEAM-135: Created by TEAM-135 (BDD scaffolding)
// TEAM-156: Added hive_catalog_steps
// TEAM-158: Added heartbeat_steps
// TEAM-159: Removed device_detection_steps - moved to rbee-hive BDD (device detection happens on hive, not queen)
// TEAM-159: Added mock_hive_device_endpoint - mocks rbee-hive's /v1/devices endpoint
// TEAM-159: Added device_storage_steps for testing device capability STORAGE (not detection)

pub mod device_storage_steps;
pub mod heartbeat_steps;
pub mod hive_catalog_steps;
pub mod integration_steps; // TEAM-159: REAL integration tests with actual daemons
pub mod mock_hive_device_endpoint;
pub mod world;
