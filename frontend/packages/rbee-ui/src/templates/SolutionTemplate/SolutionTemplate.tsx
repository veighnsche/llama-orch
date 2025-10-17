import { Card, CardContent, CardHeader, CardTitle } from "@rbee/ui/atoms";
import { FeatureInfoCard } from "@rbee/ui/molecules";
import { StepListItem } from "@rbee/ui/molecules/StepListItem";
import {
  BeeArchitecture,
  type BeeTopology,
  EarningsCard,
} from "@rbee/ui/organisms";
import { cn } from "@rbee/ui/utils";
import type { ReactNode } from "react";

export type Feature = {
  icon: ReactNode;
  title: string;
  body: string;
  badge?: string | ReactNode;
};

export type Step = {
  title: string;
  body: string;
};

export type EarningRow = {
  model: string;
  meta: string;
  value: string;
  note?: string;
};

export type Earnings = {
  title?: string;
  rows: EarningRow[];
  disclaimer?: string;
  imageSrc?: string;
};

export interface SolutionTemplateProps {
  /** Feature cards to display */
  features: Feature[];
  /** Optional steps for "How It Works" section */
  steps?: Step[];
  /** Optional earnings/metrics sidebar */
  earnings?: Earnings;
  /** Optional custom aside content (overrides earnings) */
  aside?: ReactNode;
  /** Optional BeeArchitecture topology diagram */
  topology?: BeeTopology;
  /** Custom class name */
  className?: string;
}

export function SolutionTemplate({
  features,
  steps,
  earnings,
  aside,
  topology,
  className,
}: SolutionTemplateProps) {
  return (
    <div className={cn(className)}>
      <div>
        {/* Feature Tiles */}
        <div className="mb-12 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, idx) => (
            <FeatureInfoCard
              key={idx}
              icon={feature.icon}
              title={feature.title}
              body={feature.body}
              tag={
                typeof feature.badge === "string" ? feature.badge : undefined
              }
              tone="neutral"
              size="sm"
              delay="[animation-delay:100ms]"
              className="bg-card/50"
            />
          ))}
        </div>

        {/* Optional BeeArchitecture Diagram */}
        {topology && <BeeArchitecture topology={topology} />}

        {/* Timeline + Aside (only if steps or aside/earnings provided) */}
        {(steps || aside || earnings) && (
          <div className="mt-12 grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:gap-12">
            {/* Steps Card */}
            {steps && (
              <Card className="animate-in fade-in-50 bg-card/40 [animation-delay:150ms]">
                <CardHeader>
                  <CardTitle className="text-2xl">How It Works</CardTitle>
                </CardHeader>
                <CardContent>
                  <ol className="space-y-6">
                    {steps.map((step, idx) => (
                      <StepListItem
                        key={idx}
                        number={idx + 1}
                        title={step.title}
                        body={step.body}
                      />
                    ))}
                  </ol>
                </CardContent>
              </Card>
            )}

            {/* Aside (custom or earnings) */}
            {aside ? (
              aside
            ) : earnings ? (
              <div className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
                <EarningsCard
                  title={earnings.title}
                  rows={earnings.rows}
                  disclaimer={earnings.disclaimer}
                />
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
