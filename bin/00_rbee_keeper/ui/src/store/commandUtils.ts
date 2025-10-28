// TEAM-338: Shared utilities for command execution with global isExecuting state
// Used by queenStore and hiveStore to avoid code duplication

import { useCommandStore } from "./commandStore";

/**
 * Wraps a command function with global isExecuting state management
 * Automatically sets isExecuting=true before command, false after
 * Calls refreshFn after successful command execution
 * 
 * @param commandFn - The command to execute (e.g., () => commands.queenStart())
 * @param refreshFn - Function to refresh state after command (e.g., fetchStatus)
 * @param errorContext - Context string for error logging (e.g., "Queen command")
 */
export const withCommandExecution = async (
  commandFn: () => Promise<unknown>,
  refreshFn: () => Promise<void>,
  errorContext: string = "Command",
) => {
  const { setIsExecuting } = useCommandStore.getState();
  setIsExecuting(true);
  try {
    await commandFn();
    await refreshFn();
  } catch (error) {
    console.error(`${errorContext} failed:`, error);
    throw error;
  } finally {
    setIsExecuting(false);
  }
};
