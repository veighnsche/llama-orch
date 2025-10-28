// TEAM-338: Zustand store for SSH Hives state
// Replaces SshHivesContainer with idiomatic Zustand pattern
import { create } from 'zustand';
import { commands } from '@/generated/bindings';
import type { SshTarget } from '@/generated/bindings';

export interface SshHive {
  host: string;
  host_subtitle?: string;
  hostname: string;
  user: string;
  port: number;
  status: "online" | "offline" | "unknown";
}

interface SshHivesState {
  hives: SshHive[];
  isLoading: boolean;
  error: string | null;
  
  // Actions
  fetchHives: () => Promise<void>;
  refresh: () => Promise<void>;
  reset: () => void;
}

// Convert tauri-specta SshTarget to SshHive
function convertToSshHive(target: SshTarget): SshHive {
  return {
    host: target.host,
    host_subtitle: target.host_subtitle ?? undefined,
    hostname: target.hostname,
    user: target.user,
    port: target.port,
    status: target.status,
  };
}

export const useSshHivesStore = create<SshHivesState>((set, get) => ({
  hives: [],
  isLoading: false,
  error: null,

  fetchHives: async () => {
    set({ isLoading: true, error: null });
    try {
      const result = await commands.sshList();
      if (result.status === "ok") {
        const hives = result.data.map(convertToSshHive);
        set({ hives, isLoading: false });
      } else {
        throw new Error(result.error || "Failed to load SSH hives");
      }
    } catch (error) {
      console.error("Failed to fetch SSH targets:", error);
      set({ 
        error: error instanceof Error ? error.message : 'Failed to fetch SSH targets',
        isLoading: false 
      });
    }
  },

  refresh: async () => {
    await get().fetchHives();
  },

  reset: () => {
    set({ 
      hives: [], 
      isLoading: false, 
      error: null 
    });
  },
}));
