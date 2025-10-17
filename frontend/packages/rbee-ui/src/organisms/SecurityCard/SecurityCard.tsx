import { Card, CardContent, CardFooter } from "@rbee/ui/atoms";
import { CheckItem } from "@rbee/ui/atoms/CheckItem";
import { IconCardHeader } from "@rbee/ui/molecules";
import { cn } from "@rbee/ui/utils";
import Link from "next/link";
import type * as React from "react";

export interface SecurityCardProps {
  /** Rendered icon component (e.g., <Lock className="w-6 h-6" />) */
  icon: React.ReactNode;
  /** Card title (e.g., "auth-min: Zero-Trust Authentication") */
  title: string;
  /** Optional subtitle (e.g., "The Trickster Guardians") */
  subtitle?: string;
  /** Introduction paragraph */
  intro?: string;
  /** List of security features/capabilities */
  bullets: string[];
  /** Optional documentation link */
  docsHref?: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * SecurityCard organism for displaying security capabilities
 * with consistent structure, accessibility, and optional docs link
 */
export function SecurityCard({
  icon,
  title,
  subtitle,
  intro,
  bullets,
  docsHref,
  className,
}: SecurityCardProps) {
  const titleId = `security-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

  return (
    <Card
      className={cn(
        "w-full rounded-2xl bg-card/60 transition-shadow hover:shadow-lg",
        className
      )}
      aria-labelledby={titleId}
    >
      <IconCardHeader
        icon={icon}
        title={title}
        subtitle={subtitle}
        titleId={titleId}
      />

      <CardContent>
        {/* Intro */}
        {intro && (
          <p className="mb-4 text-sm leading-relaxed text-foreground/85">
            {intro}
          </p>
        )}

        {/* Bullets */}
        <ul className="mt-2 space-y-2">
          {bullets.map((bullet, idx) => (
            <CheckItem key={idx}>{bullet}</CheckItem>
          ))}
        </ul>
      </CardContent>

      {/* Footer with optional docs link */}
      {docsHref && (
        <CardFooter className="mt-auto p-0 px-6 pb-6 pt-4">
          <Link
            href={docsHref}
            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
            aria-label={`View documentation for ${title}`}
          >
            Docs →
          </Link>
        </CardFooter>
      )}
    </Card>
  );
}
