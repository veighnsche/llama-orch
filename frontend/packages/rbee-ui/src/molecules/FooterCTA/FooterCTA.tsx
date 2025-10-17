import { Button } from "@rbee/ui/atoms";
import { cn } from "@rbee/ui/utils";
import type { ReactNode } from "react";

export interface FooterCTAProps {
  /** Optional message text */
  message?: string | ReactNode;
  /** CTA buttons */
  ctas?: Array<{
    label: string;
    href: string;
    variant?: "default" | "outline" | "ghost" | "secondary" | "destructive" | "link";
  }>;
  /** Additional CSS classes */
  className?: string;
}

export function FooterCTA({ message, ctas, className }: FooterCTAProps) {
  if (!message && (!ctas || ctas.length === 0)) {
    return null;
  }

  return (
    <div
      className={cn(
        "mt-6 flex flex-col sm:flex-row gap-3 justify-center items-center",
        className
      )}
    >
      {message && (
        <p className="text-sm text-muted-foreground text-center sm:text-left font-sans">
          {message}
        </p>
      )}
      {ctas && ctas.length > 0 && (
        <div className="flex gap-3">
          {ctas.map((cta, i) => (
            <Button
              key={i}
              asChild
              variant={cta.variant || "default"}
              size="default"
            >
              <a href={cta.href}>{cta.label}</a>
            </Button>
          ))}
        </div>
      )}
    </div>
  );
}
