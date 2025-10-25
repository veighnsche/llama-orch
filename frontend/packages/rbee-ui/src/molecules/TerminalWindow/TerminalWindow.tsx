"use client";

import { cn } from "@rbee/ui/utils";
import { Check, Copy } from "lucide-react";
import type { ReactNode } from "react";
import { useState } from "react";

export interface TerminalWindowProps {
  /** Terminal title */
  title?: string;
  /** Terminal content */
  children: ReactNode;
  /** Show terminal window chrome (traffic lights, title bar) */
  showChrome?: boolean;
  /** Optional footer content */
  footer?: ReactNode;
  /** Accessible label for screen readers */
  ariaLabel?: string;
  /** Show copy button */
  copyable?: boolean;
  /** Raw text to copy (if different from rendered children) */
  copyText?: string;
  /** Terminal variant */
  variant?: "terminal" | "code" | "output";
  /** Additional CSS classes */
  className?: string;
}

export function TerminalWindow({
  title,
  children,
  showChrome = true,
  footer,
  ariaLabel,
  copyable = false,
  copyText,
  variant = "terminal",
  className,
}: TerminalWindowProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const textToCopy =
      copyText || (typeof children === "string" ? children : "");
    if (textToCopy) {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div
      className={cn(
        "rounded border border-border bg-card overflow-hidden font-sans",
        className,
      )}
      role="region"
      aria-label={ariaLabel}
    >
      {/* Terminal top bar */}
      {showChrome && (
        <div className="flex items-center justify-between gap-1 bg-muted/50 px-4 py-2">
          <div className="flex items-center gap-1">
            <span
              className="size-2 rounded-full bg-red-500/70"
              aria-hidden="true"
            />
            <span
              className="size-2 rounded-full bg-yellow-500/70"
              aria-hidden="true"
            />
            <span
              className="size-2 rounded-full bg-green-500/70"
              aria-hidden="true"
            />
            {title && (
              <span className="ml-3 font-mono text-xs text-muted-foreground">
                {title}
              </span>
            )}
          </div>
          {copyable && (
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors rounded-md hover:bg-accent/50"
              aria-label="Copy to clipboard"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* Console content */}
      <div className="bg-background p-6 font-mono text-sm leading-relaxed">
        {children}
      </div>

      {/* Optional footer */}
      {footer && (
        <div className="border-t border-border px-6 py-3">{footer}</div>
      )}
    </div>
  );
}
