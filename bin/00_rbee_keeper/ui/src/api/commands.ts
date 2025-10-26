// TEAM-294: TypeScript API wrapper for Tauri commands
// Provides type-safe wrappers around all Tauri commands defined in tauri_commands.rs
// TEAM-296: Updated to use COMMANDS registry for type safety

import { invoke } from '@tauri-apps/api/core';
import type { CommandResponse } from './types';
import { COMMANDS } from './commands.registry';

// Helper to parse command responses
async function invokeCommand(command: string, args?: Record<string, unknown>): Promise<CommandResponse> {
  const response = await invoke<string>(command, args);
  return JSON.parse(response) as CommandResponse;
}

// ============================================================================
// STATUS COMMANDS
// ============================================================================

export async function getStatus(): Promise<CommandResponse> {
  return invokeCommand(COMMANDS.GET_STATUS);
}

// ============================================================================
// QUEEN COMMANDS
// ============================================================================

export async function queenStart(): Promise<CommandResponse> {
  return invokeCommand(COMMANDS.QUEEN_START);
}

export async function queenStop(): Promise<CommandResponse> {
  return invokeCommand(COMMANDS.QUEEN_STOP);
}

export async function queenStatus(): Promise<CommandResponse> {
  return invokeCommand(COMMANDS.QUEEN_STATUS);
}

export async function queenRebuild(withLocalHive: boolean): Promise<CommandResponse> {
  return invokeCommand(COMMANDS.QUEEN_REBUILD, { with_local_hive: withLocalHive });
}

export async function queenInfo(): Promise<CommandResponse> {
  return invokeCommand(COMMANDS.QUEEN_INFO);
}

export async function queenInstall(binary?: string): Promise<CommandResponse> {
  return invokeCommand('queen_install', { binary });
}

export async function queenUninstall(): Promise<CommandResponse> {
  return invokeCommand('queen_uninstall');
}

// ============================================================================
// HIVE COMMANDS
// ============================================================================

export async function hiveInstall(
  host: string,
  binary?: string,
  installDir?: string
): Promise<CommandResponse> {
  return invokeCommand('hive_install', {
    host,
    binary,
    install_dir: installDir,
  });
}

export async function hiveUninstall(host: string, installDir?: string): Promise<CommandResponse> {
  return invokeCommand('hive_uninstall', {
    host,
    install_dir: installDir,
  });
}

export async function hiveStart(
  host: string,
  port: number,
  installDir?: string
): Promise<CommandResponse> {
  return invokeCommand('hive_start', {
    host,
    port,
    install_dir: installDir,
  });
}

export async function hiveStop(host: string): Promise<CommandResponse> {
  return invokeCommand('hive_stop', { host });
}

export async function hiveList(): Promise<CommandResponse> {
  return invokeCommand('hive_list');
}

export async function hiveGet(alias: string): Promise<CommandResponse> {
  return invokeCommand('hive_get', { alias });
}

export async function hiveStatus(alias: string): Promise<CommandResponse> {
  return invokeCommand('hive_status', { alias });
}

export async function hiveRefreshCapabilities(alias: string): Promise<CommandResponse> {
  return invokeCommand('hive_refresh_capabilities', { alias });
}

// ============================================================================
// WORKER COMMANDS
// ============================================================================

export async function workerSpawn(
  hiveId: string,
  model: string,
  device: string
): Promise<CommandResponse> {
  return invokeCommand('worker_spawn', {
    hive_id: hiveId,
    model,
    device,
  });
}

export async function workerProcessList(hiveId: string): Promise<CommandResponse> {
  return invokeCommand('worker_process_list', { hive_id: hiveId });
}

export async function workerProcessGet(hiveId: string, pid: number): Promise<CommandResponse> {
  return invokeCommand('worker_process_get', {
    hive_id: hiveId,
    pid,
  });
}

export async function workerProcessDelete(hiveId: string, pid: number): Promise<CommandResponse> {
  return invokeCommand('worker_process_delete', {
    hive_id: hiveId,
    pid,
  });
}

// ============================================================================
// MODEL COMMANDS
// ============================================================================

export async function modelDownload(hiveId: string, model: string): Promise<CommandResponse> {
  return invokeCommand('model_download', {
    hive_id: hiveId,
    model,
  });
}

export async function modelList(hiveId: string): Promise<CommandResponse> {
  return invokeCommand('model_list', { hive_id: hiveId });
}

export async function modelGet(hiveId: string, id: string): Promise<CommandResponse> {
  return invokeCommand('model_get', {
    hive_id: hiveId,
    id,
  });
}

export async function modelDelete(hiveId: string, id: string): Promise<CommandResponse> {
  return invokeCommand('model_delete', {
    hive_id: hiveId,
    id,
  });
}

// ============================================================================
// INFERENCE COMMANDS
// ============================================================================

export async function infer(params: {
  hiveId: string;
  model: string;
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  device?: string;
  workerId?: string;
  stream?: boolean;
}): Promise<CommandResponse> {
  return invokeCommand('infer', {
    hive_id: params.hiveId,
    model: params.model,
    prompt: params.prompt,
    max_tokens: params.maxTokens,
    temperature: params.temperature,
    top_p: params.topP,
    top_k: params.topK,
    device: params.device,
    worker_id: params.workerId,
    stream: params.stream,
  });
}
