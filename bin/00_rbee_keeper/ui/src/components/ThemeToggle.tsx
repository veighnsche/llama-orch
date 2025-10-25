// TEAM-294: Tauri-compatible theme toggle
// Local version that uses our ThemeProvider instead of next-themes

import { IconButton } from "@rbee/ui/atoms/IconButton";
import { Moon, Sun } from "lucide-react";
import { useState, useEffect } from "react";
import { useTheme } from "./ThemeProvider";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Avoid hydration mismatch (not needed in Tauri, but kept for consistency)
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <IconButton aria-label="Toggle theme" title="Toggle theme">
        <Sun className="size-5" aria-hidden />
      </IconButton>
    );
  }

  const isDark = theme === "dark" || (theme === "system" && 
    window.matchMedia("(prefers-color-scheme: dark)").matches);

  return (
    <IconButton
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label="Toggle theme"
      title="Toggle theme"
    >
      {isDark ? (
        <Sun className="size-5 transition-transform duration-300" aria-hidden />
      ) : (
        <Moon className="size-5 transition-transform duration-300" aria-hidden />
      )}
    </IconButton>
  );
}
