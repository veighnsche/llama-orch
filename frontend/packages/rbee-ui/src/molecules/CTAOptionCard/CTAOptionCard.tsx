import { Badge } from "@rbee/ui/atoms/Badge";
import { cn } from "@rbee/ui/utils";
import type { ReactNode } from "react";

export interface CTAOptionCardProps {
  icon: ReactNode;
  title: string;
  body: string;
  action: ReactNode;
  tone?: "primary" | "outline";
  note?: string;
  eyebrow?: string;
  className?: string;
}

export function CTAOptionCard({
  icon,
  title,
  body,
  action,
  tone = "outline",
  note,
  eyebrow,
  className,
}: CTAOptionCardProps) {
  const titleId = `cta-option-${title.toLowerCase().replace(/\s+/g, "-")}`;
  const bodyId = `${titleId}-body`;

  return (
    <article
      role="region"
      aria-labelledby={titleId}
      aria-describedby={bodyId}
      className={cn(
        // Base structure & surface
        "group relative h-full flex flex-col rounded-2xl border border-border/70 bg-card/70 p-6 sm:p-7 backdrop-blur-sm shadow-sm",
        // Entrance animation
        "animate-in fade-in-50 zoom-in-95 duration-300",
        // Interactive depth
        "hover:border-primary/40 hover:shadow-md focus-within:shadow-md transition-shadow",
        // Focus ring for keyboard users
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2",
        // Primary tone overrides
        tone === "primary" && "border-primary/40 bg-primary/5",
        className
      )}
    >
      {/* Subtle radial highlight for primary tone */}
      {tone === "primary" && (
        <span
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-8 -top-6 h-20 rounded-full bg-primary/10 blur-2xl"
        />
      )}

      {/* Header: Icon chip + eyebrow */}
      <header className="mb-4 flex flex-col items-center">
        {/* Icon chip with halo */}
        <div className="relative">
          {/* Halo ring */}
          <span
            aria-hidden="true"
            className="absolute inset-0 rounded-2xl ring-1 ring-primary/10"
          />
          {/* Icon chip */}
          <div
            className="relative rounded-xl bg-primary/12 text-primary p-3"
            aria-hidden="true"
          >
            {icon}
          </div>
        </div>

        {/* Eyebrow label */}
        {eyebrow && (
          <Badge
            variant="outline"
            className="mt-2 bg-primary/10 border-primary/30 text-primary text-[11px] font-medium"
          >
            {eyebrow}
          </Badge>
        )}
      </header>

      {/* Content: Title + body */}
      <div role="doc-subtitle" className="mb-5">
        <h3
          id={titleId}
          className={cn(
            "mb-2 text-center text-2xl font-semibold tracking-tight",
            tone === "primary" ? "text-primary" : "text-foreground"
          )}
        >
          {title}
        </h3>

        <p
          id={bodyId}
          className="font-sans text-center text-sm leading-6 text-muted-foreground max-w-[80ch] mx-auto"
        >
          {body}
        </p>
      </div>

      {/* Footer: Primary action + note */}
      <footer className="mt-auto">
        {action}
        {note && (
          <p className="mt-2 text-center font-sans text-[11px] text-muted-foreground">
            {note}
          </p>
        )}
      </footer>
    </article>
  );
}
