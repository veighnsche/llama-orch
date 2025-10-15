import { cn } from "@rbee/ui/utils";
import type { ReactNode } from "react";

export interface TerminalWindowProps {
  /** Terminal title */
  title?: string;
  /** Terminal content */
  children: ReactNode;
  /** Terminal variant */
  variant?: "terminal" | "code" | "output";
  /** Additional CSS classes */
  className?: string;
}

export function TerminalWindow({
  title,
  children,
  variant = "terminal",
  className,
}: TerminalWindowProps) {
  return (
    <div
      className={cn(
        "bg-card border border-border rounded-lg overflow-hidden shadow-2xl",
        className
      )}
    >
      <div className="flex items-center gap-2 px-4 py-3 bg-muted border-b border-border">
        <div className="flex gap-2" aria-hidden="true">
          <div
            className="h-3 w-3 rounded-full"
            style={{ backgroundColor: "var(--terminal-red)" }}
          ></div>
          <div
            className="h-3 w-3 rounded-full"
            style={{ backgroundColor: "var(--terminal-amber)" }}
          ></div>
          <div
            className="h-3 w-3 rounded-full"
            style={{ backgroundColor: "var(--terminal-green)" }}
          ></div>
        </div>
        {title && (
          <span className="text-muted-foreground text-sm ml-2 font-mono">
            {title}
          </span>
        )}
      </div>
      <div className="p-6 text-sm font-mono">{children}</div>
    </div>
  );
}
