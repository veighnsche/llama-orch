export interface HelperTextItemProps {
  /** Title/label for the helper text */
  title: string
  /** Description text */
  description: string
}

/**
 * HelperTextItem - Inline title + description for page footer help
 * 
 * Displays a muted helper text with an emphasized title inline with the description.
 * Used in PageContainer's helperText prop.
 * 
 * @example
 * ```tsx
 * <HelperTextItem
 *   title="Queen"
 *   description="routes inference jobs to the right worker in the right hive. Start Queen first to enable job routing."
 * />
 * ```
 */
export function HelperTextItem({ title, description }: HelperTextItemProps) {
  return (
    <p className="text-xs text-muted-foreground/70">
      <span className="font-medium text-muted-foreground">{title}</span> {description}
    </p>
  )
}
