import { Button } from "@rbee/ui/atoms/Button";
import { cn } from "@rbee/ui/utils";
import { cva, type VariantProps } from "class-variance-authority";
import { ArrowRight } from "lucide-react";
import * as React from "react";

const buttonCardFooterVariants = cva(
  "sticky bottom-0 z-10 flex flex-col gap-3",
  {
    variants: {
      variant: {
        default: "",
        elevated: "shadow-lg",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

const buttonVariants = cva("w-full", {
  variants: {
    color: {
      primary: "bg-primary",
      "chart-1": "bg-chart-1",
      "chart-2": "bg-chart-2",
      "chart-3": "bg-chart-3",
      "chart-4": "bg-chart-4",
      "chart-5": "bg-chart-5",
    },
  },
  defaultVariants: {
    color: "primary",
  },
});

export interface ButtonCardFooterProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, "children">,
    VariantProps<typeof buttonCardFooterVariants> {
  /** Optional badge slot above the button */
  badgeSlot?: React.ReactNode;
  /** Button text */
  buttonText: string;
  /** Button href */
  href: string;
  /** Button color variant */
  buttonColor?: VariantProps<typeof buttonVariants>["color"];
  /** Optional aria-describedby for accessibility */
  ariaDescribedBy?: string;
  /** Show arrow icon */
  showArrow?: boolean;
}

const ButtonCardFooter = React.forwardRef<
  HTMLDivElement,
  ButtonCardFooterProps
>(
  (
    {
      className,
      variant,
      badgeSlot,
      buttonText,
      href,
      buttonColor = "primary",
      ariaDescribedBy,
      showArrow = true,
      ...props
    },
    ref
  ) => {
    return (
      <div
        ref={ref}
        data-slot="card-footer"
        className={cn(buttonCardFooterVariants({ variant }), className)}
        {...props}
      >
        {badgeSlot}
        <a href={href} className="w-full">
          <Button
            className={buttonVariants({ color: buttonColor })}
            aria-describedby={ariaDescribedBy}
          >
            {buttonText}
            {showArrow && (
              <ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
            )}
          </Button>
        </a>
      </div>
    );
  }
);
ButtonCardFooter.displayName = "ButtonCardFooter";

export { ButtonCardFooter };
