import { CodeExamplesSection } from '@rbee/ui/organisms'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface CodeExample {
  id: string
  title: string
  summary: string
  language: string
  code: string
}

export interface DevelopersCodeExamplesProps {
  /** Section title */
  title: string
  /** Section subtitle */
  subtitle: string
  /** Footer note */
  footerNote: string
  /** Code examples */
  items: CodeExample[]
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * DevelopersCodeExamples - Code examples section for developers page
 *
 * @example
 * ```tsx
 * <DevelopersCodeExamplesTemplate
 *   title="Build AI agents with llama-orch-utils"
 *   subtitle="TypeScript utilities for LLM pipelines and agentic workflows."
 *   footerNote="Works with any OpenAI-compatible client."
 *   items={[
 *     {
 *       id: 'simple',
 *       title: 'Simple code generation',
 *       summary: 'Invoke to generate a TypeScript validator.',
 *       language: 'TypeScript',
 *       code: `import { invoke } from '@llama-orch/utils';...`
 *     }
 *   ]}
 * />
 * ```
 */
export function DevelopersCodeExamplesTemplate({ title, subtitle, footerNote, items }: DevelopersCodeExamplesProps) {
  return <CodeExamplesSection title={title} subtitle={subtitle} footerNote={footerNote} items={items} />
}
