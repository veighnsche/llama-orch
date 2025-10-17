import { Card } from "@rbee/ui/atoms";
import { Disclaimer, GPUListItem, IconCardHeader } from "@rbee/ui/molecules";
import { TrendingUp } from "lucide-react";

export type EarningRow = {
  model: string;
  meta: string;
  value: string;
  note?: string;
};

export interface EarningsCardProps {
  /** Card title */
  title?: string;
  /** Earnings rows */
  rows: EarningRow[];
  /** Optional disclaimer text */
  disclaimer?: string;
  /** Custom class name */
  className?: string;
}

export function EarningsCard({
  title = "Compliance Metrics",
  rows,
  disclaimer,
  className,
}: EarningsCardProps) {
  return (
    <Card className={className}>
      <IconCardHeader
        icon={<TrendingUp />}
        title={title}
        iconSize="sm"
        iconTone="success"
        titleClassName="font-semibold"
        align="center"
      />
      <div className="px-6 pb-6 space-y-2">
        {rows.map((row, idx) => (
          <GPUListItem
            key={idx}
            name={row.model}
            subtitle={row.meta}
            value={row.value}
            label={row.note}
            className="border-0 bg-transparent p-0"
          />
        ))}
      </div>

      {disclaimer && (
        <div className="px-6 pb-6">
          <Disclaimer variant="card">{disclaimer}</Disclaimer>
        </div>
      )}
    </Card>
  );
}
