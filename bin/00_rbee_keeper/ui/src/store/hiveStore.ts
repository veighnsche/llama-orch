// Zustand store for Hive UI preferences (persistent)
// Service states (on/off) come from backend via heartbeats, not stored here
import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface SshTarget {
  id: string;
  name: string;
  user: string;
  hostname: string;
  port: number;
}

interface HiveStore {
  // Currently selected SSH target for Hive operations
  selectedTarget: string;
  setSelectedTarget: (target: string) => void;

  // User's favorite SSH targets (for quick access)
  favoriteTargets: string[];
  addFavorite: (targetId: string) => void;
  removeFavorite: (targetId: string) => void;
  isFavorite: (targetId: string) => boolean;
}

export const useHiveStore = create<HiveStore>()(
  persist(
    (set, get) => ({
      // Default to localhost
      selectedTarget: "localhost",
      setSelectedTarget: (target) => set({ selectedTarget: target }),

      // Empty favorites by default
      favoriteTargets: [],
      addFavorite: (targetId) =>
        set((state) => ({
          favoriteTargets: [...new Set([...state.favoriteTargets, targetId])],
        })),
      removeFavorite: (targetId) =>
        set((state) => ({
          favoriteTargets: state.favoriteTargets.filter((id) => id !== targetId),
        })),
      isFavorite: (targetId) => get().favoriteTargets.includes(targetId),
    }),
    {
      name: "hive-preferences", // localStorage key
    }
  )
);
