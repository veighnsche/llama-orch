import { Badge } from '@rbee/ui/atoms/Badge'
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from '@rbee/ui/atoms/Table'
import { Tooltip, TooltipContent, TooltipTrigger } from '@rbee/ui/atoms/Tooltip'
import { cn } from '@rbee/ui/utils'
import { Check, X } from 'lucide-react'
import React from 'react'

export interface Provider {
  key: string
  label: string
  /** Optional subtitle/description for the column */
  subtitle?: string
  /** Optional badge to display in column header */
  badge?: string
  /** Highlight/emphasize this column */
  accent?: boolean
}

export interface Row {
  feature: string
  values: Record<string, boolean | 'Partial' | string>
  /** Optional tooltip note for the feature */
  note?: string
  /** Optional group identifier for grouping rows */
  group?: string
}

export interface RowGroup {
  /** Group identifier */
  id: string
  /** Group label to display */
  label: string
}

export interface MatrixTableProps {
  columns: Provider[]
  rows: Row[]
  /** Optional groups for organizing rows */
  groups?: RowGroup[]
  /** Optional custom caption (defaults to generic message) */
  caption?: string
  className?: string
}

export function MatrixTable({ columns, rows, groups, caption, className }: MatrixTableProps) {
  // Group rows if groups are provided
  const groupedRows = groups
    ? groups.reduce(
        (acc, group) => {
          acc[group.id] = rows.filter((r) => r.group === group.id)
          return acc
        },
        {} as Record<string, Row[]>,
      )
    : null

  const colSpan = columns.length + 1 // +1 for the feature column
  const renderCell = (value: boolean | 'Partial' | string, _providerKey: string) => {
    if (value === true) {
      return <Check className="mx-auto h-5 w-5 text-chart-3" aria-label="Included" />
    }
    if (value === false) {
      return <X className="mx-auto h-5 w-5 text-destructive" aria-label="Not available" />
    }
    if (value === 'Partial') {
      return (
        <span
          className="inline-flex rounded-full border/60 bg-background px-2 py-0.5 text-xs text-foreground/80"
          aria-label="Partial"
          title="Available with constraints (region, SKU, or config)"
        >
          Partial
        </span>
      )
    }
    return <span className="text-xs text-foreground/80">{value}</span>
  }

  const renderFeatureLabel = (row: Row) => {
    if (row.note) {
      return (
        <Tooltip>
          <TooltipTrigger asChild>
            <button className="text-left underline decoration-dotted underline-offset-4 hover:text-foreground transition-colors">
              {row.feature}
            </button>
          </TooltipTrigger>
          <TooltipContent>{row.note}</TooltipContent>
        </Tooltip>
      )
    }
    return row.feature
  }

  const renderRowGroupLabel = (label: string) => (
    <TableRow className="bg-secondary/50">
      <TableHead colSpan={colSpan} className="p-3 text-left text-sm font-semibold text-foreground">
        {label}
      </TableHead>
    </TableRow>
  )

  const renderRow = (row: Row, index: number) => (
    <TableRow
      key={index}
      className="border-b border-border/80 transition-colors hover:bg-secondary/30 odd:bg-background even:bg-background/60"
    >
      <TableHead className="p-3 text-left text-sm font-normal text-muted-foreground sticky left-0 z-10 bg-card">
        {renderFeatureLabel(row)}
      </TableHead>
      {columns.map((col) => (
        <TableCell key={col.key} className={cn('p-3 text-center', col.accent && 'bg-primary/5')}>
          {renderCell(row.values[col.key], col.key)}
        </TableCell>
      ))}
    </TableRow>
  )

  return (
    <Table className={cn('table-fixed', className)}>
      <TableCaption className="sr-only">{caption || 'Feature comparison table'}</TableCaption>
      <colgroup>
        <col className="w-1/3" />
        {columns.map((col) => (
          <col key={col.key} />
        ))}
      </colgroup>
      <TableHeader>
        <TableRow className="border-b border-border/80">
          <TableHead className="p-3 text-left text-xs uppercase tracking-wide text-muted-foreground sticky left-0 z-10 bg-card">
            Feature
          </TableHead>
          {columns.map((col) => (
            <TableHead
              key={col.key}
              className={cn('p-3 text-center', col.accent && 'bg-primary/5 ring-1 ring-primary/10')}
            >
              <div
                className={cn(
                  'text-xs uppercase tracking-wide font-semibold',
                  col.accent ? 'text-primary' : 'text-muted-foreground',
                )}
              >
                {col.label}
              </div>
              {col.subtitle && <div className="text-xs text-muted-foreground font-normal mt-0.5">{col.subtitle}</div>}
              {col.badge && (
                <Badge variant="outline" className="mt-1 rounded-full text-xs animate-in fade-in">
                  {col.badge}
                </Badge>
              )}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {groupedRows && groups
          ? // Render grouped rows
            groups.map((group) => (
              <React.Fragment key={group.id}>
                {renderRowGroupLabel(group.label)}
                {groupedRows[group.id]?.map((row, rowIndex) => renderRow(row, rowIndex))}
              </React.Fragment>
            ))
          : // Render ungrouped rows
            rows.map((row, i) => renderRow(row, i))}
      </TableBody>
    </Table>
  )
}
