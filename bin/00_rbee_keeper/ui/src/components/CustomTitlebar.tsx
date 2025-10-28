// TEAM-334: Custom titlebar for rbee-keeper
// Replaces native window decorations with React component
// Provides window controls (minimize, maximize, close) and drag area

import { getCurrentWindow } from "@tauri-apps/api/window";
import { X, Minus, Square } from "lucide-react";
import { Button } from "@rbee/ui/atoms";
import { BrandLogo } from "@rbee/ui/molecules";

export function CustomTitlebar() {
  const appWindow = getCurrentWindow();

  const handleMinimize = () => {
    appWindow.minimize();
  };

  const handleMaximize = () => {
    appWindow.toggleMaximize();
  };

  const handleClose = () => {
    appWindow.close();
  };

  return (
    <div
      data-tauri-drag-region
      className="h-10 bg-background border-b border-border flex items-center justify-between px-3 select-none"
    >
      {/* Left side - Brand logo */}
      <div className="flex items-center">
        <BrandLogo size="sm" />
      </div>

      {/* Right side - Window controls */}
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleMinimize}
          aria-label="Minimize"
        >
          <Minus />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleMaximize}
          aria-label="Maximize"
        >
          <Square className="h-3 w-3" />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={handleClose}
          aria-label="Close"
          className="hover:bg-destructive hover:text-destructive-foreground"
        >
          <X />
        </Button>
      </div>
    </div>
  );
}
