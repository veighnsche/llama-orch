import { Slot } from "@radix-ui/react-slot";
import { cn } from "@rbee/ui/utils";
import {
  brandLink,
  focusRing,
  focusRingDestructive,
} from "@rbee/ui/utils/focus-ring";
import { cva, type VariantProps } from "class-variance-authority";
import type * as React from "react";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded text-sm font-sans font-medium transition-[transform,background-color,box-shadow] duration-150 disabled:pointer-events-none disabled:bg-slate-200 disabled:text-slate-400 disabled:border-slate-300 dark:disabled:bg-[#1a2435] dark:disabled:text-[#6c7a90] dark:disabled:border-[#223047] [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        default: cn(
          "bg-primary text-primary-foreground hover:bg-accent active:bg-[#92400e] active:scale-[0.98]",
          "dark:bg-[#b45309] dark:hover:bg-[#d97706] dark:active:bg-[#92400e]",
          focusRing,
        ),
        destructive: cn(
          "bg-destructive text-destructive-foreground hover:bg-destructive/90 active:bg-destructive/80 active:scale-[0.98] dark:bg-destructive/60",
          focusRingDestructive,
        ),
        outline: cn(
          "border border-border bg-transparent text-foreground shadow-xs hover:bg-[#f4f6f9] hover:border-slate-400 active:bg-[#eef2f6]",
          "dark:bg-input/30 dark:border-input dark:hover:bg-input/50 dark:hover:text-foreground",
          focusRing,
        ),
        secondary: cn(
          "bg-secondary text-secondary-foreground hover:bg-[#f4f6f9] active:bg-[#eef2f6] border border-transparent",
          focusRing,
        ),
        ghost: cn(
          "bg-transparent text-foreground hover:bg-[#f4f6f9] active:bg-[#eef2f6]",
          "dark:hover:bg-white/[0.04] dark:active:bg-white/[0.06]",
          focusRing,
        ),
        link: brandLink,
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        sm: "h-8 gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 px-6 has-[>svg]:px-4",
        icon: "size-9",
        "icon-sm": "size-8",
        "icon-lg": "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

type ButtonProps = React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  };

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: ButtonProps) {
  const Comp = asChild ? Slot : "button";

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { Button, buttonVariants, type ButtonProps };
