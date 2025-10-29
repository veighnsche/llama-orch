// AUTO-GENERATED from frontend/packages/shared-config/src/ports.ts
// DO NOT EDIT MANUALLY - Run 'pnpm generate:rust' in shared-config package to update
// 
// This file provides port constants for Rust build.rs scripts
// Last generated: 2025-10-29T21:02:10.455Z

// TEAM-351: Shared port configuration constants
// TEAM-351: Bug fixes - Validation, error handling, null port comments

pub const KEEPER_DEV_PORT: u16 = 5173;
// KEEPER_PROD_PORT is null (no HTTP port)

pub const QUEEN_DEV_PORT: u16 = 7834;
pub const QUEEN_PROD_PORT: u16 = 7833;
pub const QUEEN_BACKEND_PORT: u16 = 7833;

pub const HIVE_DEV_PORT: u16 = 7836;
pub const HIVE_PROD_PORT: u16 = 7835;
pub const HIVE_BACKEND_PORT: u16 = 7835;

pub const WORKER_DEV_PORT: u16 = 7837;
pub const WORKER_PROD_PORT: u16 = 8080;
pub const WORKER_BACKEND_PORT: u16 = 8080;
