// TEAM-294: TypeScript API wrapper for Tauri commands
// Provides type-safe wrappers around all Tauri commands defined in tauri_commands.rs
// TEAM-296: Updated to use COMMANDS registry for type safety
// TEAM-334: Cleaned up - only ssh_list command remains (rest removed with tauri commands cleanup)

import { commands } from '../generated/bindings';
export type { SshTarget, SshTargetStatus } from '../generated/bindings';

// ============================================================================
// SSH COMMANDS
// ============================================================================

export async function sshList() {
  return commands.sshList();
}

// ============================================================================
// DEPRECATED - Commands removed in TEAM-334 cleanup
// ============================================================================
// All other commands have been removed from tauri_commands.rs
// They will be re-implemented later when the architecture stabilizes
// For now, only ssh_list is available
