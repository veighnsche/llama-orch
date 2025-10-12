import { cn } from '@/lib/utils'
import { Check, X } from 'lucide-react'
import type { ReactNode } from 'react'

export interface ComparisonTableRowProps {
  feature: string
  values: Array<boolean | string | ReactNode>
  highlightColumn?: number
  className?: string
}

export function ComparisonTableRow({
  feature,
  values,
  highlightColumn,
  className,
}: ComparisonTableRowProps) {
  const renderValue = (value: boolean | string | ReactNode, index: number) => {
    if (typeof value === 'boolean') {
      return value ? (
        <Check className="h-5 w-5 text-chart-3 mx-auto" />
      ) : (
        <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
      )
    }
    return value
  }

  return (
    <tr className={cn('border-b border-border', className)}>
      <td className="p-4 text-muted-foreground">{feature}</td>
      {values.map((value, index) => (
        <td
          key={index}
          className={cn(
            'text-center p-4',
            highlightColumn === index && 'bg-primary/5'
          )}
        >
          {renderValue(value, index)}
        </td>
      ))}
    </tr>
  )
}
