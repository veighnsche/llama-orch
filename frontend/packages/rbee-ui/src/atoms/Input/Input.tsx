import { cn } from '@rbee/ui/utils'
import { focusRing } from '@rbee/ui/utils/focus-ring'
import type * as React from 'react'

function Input({ className, type, ...props }: React.ComponentProps<'input'>) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        'file:text-foreground placeholder:text-slate-400 dark:placeholder:text-[#8b9bb0] selection:bg-primary selection:text-primary-foreground bg-white dark:bg-[color:var(--background)] border-input h-9 w-full min-w-0 rounded border px-3 py-1 text-base transition-[color,box-shadow,border-color] file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:cursor-not-allowed md:text-sm',
        '[box-shadow:inset_0_1px_0_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)]',
        'dark:[box-shadow:inset_0_1px_0_rgba(255,255,255,0.04),0_1px_2px_rgba(0,0,0,0.25)]',
        'hover:border-slate-400',
        'focus-visible:border-ring',
        'disabled:bg-[#1a2435] disabled:text-[#6c7a90] disabled:border-[#223047] disabled:placeholder:text-[#6c7a90]',
        focusRing,
        'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
        className,
      )}
      {...props}
    />
  )
}

export { Input }
