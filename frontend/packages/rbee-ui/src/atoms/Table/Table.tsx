'use client'

import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

function Table({ className, ...props }: React.ComponentProps<'table'>) {
  return (
    <div data-slot="table-container" className="relative w-full overflow-x-auto">
      <table data-slot="table" className={cn('w-full caption-bottom text-sm', className)} {...props} />
    </div>
  )
}

function TableHeader({ className, ...props }: React.ComponentProps<'thead'>) {
  return <thead data-slot="table-header" className={cn('[&_tr]:border-b', className)} {...props} />
}

function TableBody({ className, ...props }: React.ComponentProps<'tbody'>) {
  return <tbody data-slot="table-body" className={cn('[&_tr:last-child]:border-0', className)} {...props} />
}

function TableFooter({ className, ...props }: React.ComponentProps<'tfoot'>) {
  return (
    <tfoot
      data-slot="table-footer"
      className={cn(
        'bg-[rgba(2,6,23,0.02)] dark:bg-[rgba(255,255,255,0.03)] border-t font-medium [&>tr]:last:border-b-0',
        className
      )}
      {...props}
    />
  )
}

function TableRow({ className, ...props }: React.ComponentProps<'tr'>) {
  return (
    <tr
      data-slot="table-row"
      className={cn(
        'hover:bg-[rgba(2,6,23,0.03)] dark:hover:bg-[rgba(255,255,255,0.025)] data-[state=selected]:bg-[rgba(2,6,23,0.04)] dark:data-[state=selected]:bg-[rgba(255,255,255,0.04)] border-b border-border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--ring)] focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--background)]',
        className,
      )}
      {...props}
    />
  )
}

function TableHead({ className, sticky, ...props }: React.ComponentProps<'th'> & { sticky?: boolean }) {
  return (
    <th
      data-slot="table-head"
      data-sticky={sticky}
      className={cn(
        'text-slate-700 dark:text-slate-200 bg-[rgba(2,6,23,0.02)] dark:bg-[rgba(255,255,255,0.03)] h-10 px-2 text-left align-middle font-medium whitespace-nowrap [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]',
        'data-[sticky]:sticky data-[sticky]:top-0 data-[sticky]:z-10 data-[sticky]:backdrop-blur-[2px] data-[sticky]:dark:bg-[rgba(20,28,42,0.85)]',
        className,
      )}
      {...props}
    />
  )
}

function TableCell({ className, ...props }: React.ComponentProps<'td'>) {
  return (
    <td
      data-slot="table-cell"
      className={cn(
        'p-2 align-middle whitespace-nowrap tabular-nums dark:text-slate-200 [&:has([role=checkbox])]:pr-0 [&>[role=checkbox]]:translate-y-[2px]',
        className,
      )}
      {...props}
    />
  )
}

function TableCaption({ className, ...props }: React.ComponentProps<'caption'>) {
  return (
    <caption data-slot="table-caption" className={cn('text-muted-foreground mt-4 text-sm', className)} {...props} />
  )
}

export { Table, TableHeader, TableBody, TableFooter, TableHead, TableRow, TableCell, TableCaption }
