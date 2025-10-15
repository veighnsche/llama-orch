import { SectionContainer, UseCaseCard } from "@rbee/ui/molecules";
import { cn } from "@rbee/ui/utils";
import type { LucideIcon } from "lucide-react";

export type UseCase = {
  icon: LucideIcon;
  title: string;
  scenario: string;
  solution: string;
  outcome: string;
  tags?: string[];
  cta?: { label: string; href: string };
  illustrationSrc?: string;
};

export type UseCasesSectionProps = {
  title: string;
  subtitle?: string;
  items: UseCase[];
  columns?: 2 | 3;
  id?: string;
  className?: string;
};

export function UseCasesSection({
  title,
  subtitle,
  items,
  columns = 3,
  id,
  className,
}: UseCasesSectionProps) {
  const gridCols =
    columns === 2 ? "sm:grid-cols-2" : "sm:grid-cols-2 lg:grid-cols-3";

  return (
    <SectionContainer
      title={title}
      description={subtitle}
      bgVariant="secondary"
      paddingY="2xl"
      maxWidth="6xl"
      align="center"
      className={cn("border-b border-border", className)}
      headingId={id}
    >
      {/* Cards grid */}
      <div
        className={cn(
          "mx-auto grid max-w-6xl gap-6 animate-in fade-in-50 duration-400",
          gridCols
        )}
      >
        {items.map((item, i) => (
          <UseCaseCard
            key={i}
            icon={item.icon}
            title={item.title}
            scenario={item.scenario}
            solution={item.solution}
            outcome={item.outcome}
            tags={item.tags}
            cta={item.cta}
            iconSize="md"
            iconTone="primary"
            style={{ animationDelay: `${i * 80}ms` }}
            className="animate-in fade-in slide-in-from-bottom-2 duration-400"
          />
        ))}
      </div>
    </SectionContainer>
  );
}
