'use client'

import { CardDescription, CardTitle } from '@rbee/ui/atoms/Card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@rbee/ui/atoms/Tabs'
import { CodeBlock } from '@rbee/ui/molecules/CodeBlock'
import { cn } from '@rbee/ui/utils'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface CodeExample {
  id: string
  title: string
  summary?: string
  language?: string
  code: string
  badge?: string
}

export interface CodeExamplesTemplateProps {
  /** Code examples */
  items: CodeExample[]
  /** Footer note */
  footerNote?: string
  /** Default active example ID */
  defaultId?: string
  /** Additional CSS classes */
  className?: string
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * CodeExamplesTemplate - Interactive code examples with tabbed navigation
 *
 * @example
 * ```tsx
 * <CodeExamplesTemplate
 *   items={[
 *     {
 *       id: 'simple',
 *       title: 'Simple code generation',
 *       summary: 'Invoke to generate a TypeScript validator.',
 *       language: 'TypeScript',
 *       code: `import { invoke } from '@llama-orch/utils';...`
 *     }
 *   ]}
 *   footerNote="Works with any OpenAI-compatible client."
 * />
 * ```
 */
export function CodeExamplesTemplate({ items, defaultId, className, footerNote }: CodeExamplesTemplateProps) {
  return (
    <Tabs defaultValue={defaultId || items[0]?.id} className={cn('', className)}>
      {/* Two-column layout */}
      <div className="mx-auto w-full max-w-6xl lg:grid lg:grid-cols-12 lg:gap-10 xl:gap-12">
        {/* Left rail: example list */}
        <div className="lg:col-span-5">
          <TabsList aria-label="Code examples" className="space-y-3">
            {items.map((item, i) => (
              <TabsTrigger
                key={item.id}
                value={item.id}
                style={{ animationDelay: `${i * 80}ms` }}
                className={cn(
                  'w-full p-4 animate-in fade-in slide-in-from-bottom-2 duration-400',
                  'data-[state=active]:border-primary data-[state=active]:bg-primary/5',
                  'data-[state=inactive]:border-border/70 data-[state=inactive]:bg-card',
                )}
              >
                <div className="flex flex-col items-start gap-1">
                  <CardTitle className="text-base">{item.title}</CardTitle>
                  {item.summary && <CardDescription>{item.summary}</CardDescription>}
                </div>
              </TabsTrigger>
            ))}
          </TabsList>
        </div>

        {/* Right preview: sticky code panel */}
        <div className="mt-8 lg:col-span-7 lg:mt-0">
          <div className="lg:sticky lg:top-24">
            {items.map((item) => (
              <TabsContent key={item.id} value={item.id} className="animate-in fade-in-50 duration-200">
                <CodeBlock
                  code={item.code}
                  language={item.language}
                  title={item.badge}
                  copyable={true}
                  className="w-full min-w-0"
                />
              </TabsContent>
            ))}

            {/* Optional footer note */}
            {footerNote && <p className="mt-4 text-center text-sm text-muted-foreground font-sans">{footerNote}</p>}
          </div>
        </div>
      </div>
    </Tabs>
  )
}
