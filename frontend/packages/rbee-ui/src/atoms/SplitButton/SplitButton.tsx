// Split Button - Fused button with primary action + dropdown menu
// Primary button takes most space, dropdown trigger is square (1:1) with icon only

import { ChevronDown } from 'lucide-react'
import type * as React from 'react'
import { Button, type ButtonProps } from '../Button'
import { ButtonGroup, ButtonGroupSeparator } from '../ButtonGroup'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '../DropdownMenu'

export interface SplitButtonProps {
  /** Primary button text */
  children: React.ReactNode
  /** Primary button click handler */
  onClick?: () => void
  /** Primary button variant */
  variant?: ButtonProps['variant']
  /** Primary button size */
  size?: ButtonProps['size']
  /** Primary button icon (optional) */
  icon?: React.ReactNode
  /** Dropdown menu content */
  dropdownContent: React.ReactNode
  /** Disabled state */
  disabled?: boolean
  /** Additional className for the container */
  className?: string
}

export function SplitButton({
  children,
  onClick,
  variant = 'default',
  size = 'default',
  icon,
  dropdownContent,
  disabled = false,
  className,
}: SplitButtonProps) {
  return (
    <ButtonGroup className={className}>
      {/* Primary Action Button */}
      <Button variant={variant} size={size} onClick={onClick} disabled={disabled} className="flex-1 rounded-r-none">
        {icon && icon}
        {children}
      </Button>

      {/* Separator */}
      <ButtonGroupSeparator />

      {/* Dropdown Trigger (Square, Icon Only) */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant={variant}
            size={size}
            disabled={disabled}
            className="rounded-l-none px-2 aspect-square"
            aria-label="More actions"
          >
            <ChevronDown className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">{dropdownContent}</DropdownMenuContent>
      </DropdownMenu>
    </ButtonGroup>
  )
}
