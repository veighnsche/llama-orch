// Zustand store for installation state (persistent)
// Service states (on/off) come from backend via heartbeats, not stored here
import { create } from "zustand";
import { persist } from "zustand/middleware";

interface InstallationStore {
  // Queen installation state
  isQueenInstalled: boolean;
  setQueenInstalled: (installed: boolean) => void;

  // Installed hives (array of SSH target IDs where hive is installed)
  installedHives: string[];
  addInstalledHive: (targetId: string) => void;
  removeInstalledHive: (targetId: string) => void;
  isHiveInstalled: (targetId: string) => boolean;
}

export const useInstallationStore = create<InstallationStore>()(
  persist(
    (set, get) => ({
      // Queen not installed by default
      isQueenInstalled: false,
      setQueenInstalled: (installed) => set({ isQueenInstalled: installed }),

      // No hives installed by default
      installedHives: [],
      addInstalledHive: (targetId) =>
        set((state) => ({
          installedHives: [...new Set([...state.installedHives, targetId])],
        })),
      removeInstalledHive: (targetId) =>
        set((state) => ({
          installedHives: state.installedHives.filter((id) => id !== targetId),
        })),
      isHiveInstalled: (targetId) => get().installedHives.includes(targetId),
    }),
    {
      name: "installation-state", // localStorage key
    }
  )
);
