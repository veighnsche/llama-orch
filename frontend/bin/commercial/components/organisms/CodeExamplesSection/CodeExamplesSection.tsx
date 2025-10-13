'use client'

import { useState, type KeyboardEvent } from 'react'
import { cn } from '@/lib/utils'

export type ExampleItem = {
  id: string
  title: string
  summary?: string
  language?: string
  code: string
  badge?: string
}

export type CodeExamplesSectionProps = {
  title: string
  subtitle?: string
  items: ExampleItem[]
  defaultId?: string
  id?: string
  className?: string
  footerNote?: string
}

export function CodeExamplesSection({
  title,
  subtitle,
  items,
  defaultId,
  id,
  className,
  footerNote,
}: CodeExamplesSectionProps) {
  const [activeId, setActiveId] = useState(defaultId || items[0]?.id)
  const [copied, setCopied] = useState(false)

  const activeExample = items.find((item) => item.id === activeId) || items[0]

  const handleCopy = async () => {
    if (activeExample?.code) {
      await navigator.clipboard.writeText(activeExample.code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLButtonElement>, currentId: string) => {
    const currentIndex = items.findIndex((item) => item.id === currentId)
    let nextIndex = currentIndex

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      nextIndex = (currentIndex + 1) % items.length
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      nextIndex = currentIndex === 0 ? items.length - 1 : currentIndex - 1
    } else if (e.key === 'Home') {
      e.preventDefault()
      nextIndex = 0
    } else if (e.key === 'End') {
      e.preventDefault()
      nextIndex = items.length - 1
    }

    if (nextIndex !== currentIndex) {
      const nextId = items[nextIndex]?.id
      if (nextId) {
        setActiveId(nextId)
      }
    }
  }

  return (
    <section id={id} className={cn('border-b border-border py-24', className)}>
      <div className="container mx-auto px-4 animate-in fade-in-50 duration-400">
        {/* Heading block */}
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">{title}</h2>
          {subtitle && <p className="text-balance text-lg leading-relaxed text-muted-foreground">{subtitle}</p>}
        </div>

        {/* Two-column layout */}
        <div className="mx-auto mt-16 max-w-6xl lg:grid lg:grid-cols-12 lg:gap-10 xl:gap-12">
          {/* Left rail: example list */}
          <div className="lg:col-span-5">
            <div role="tablist" aria-label={title} className="space-y-3">
              {items.map((item, i) => (
                <button
                  key={item.id}
                  role="tab"
                  aria-selected={activeId === item.id}
                  aria-controls={`panel-${item.id}`}
                  tabIndex={activeId === item.id ? 0 : -1}
                  onClick={() => setActiveId(item.id)}
                  onKeyDown={(e) => handleKeyDown(e, item.id)}
                  style={{ animationDelay: `${i * 80}ms` }}
                  className={cn(
                    'w-full rounded-xl border p-4 text-left transition-all',
                    'hover:border-primary/40 hover:bg-card/80',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40',
                    'animate-in fade-in slide-in-from-bottom-2 duration-400',
                    activeId === item.id
                      ? 'border-primary bg-primary/5'
                      : 'border-border/70 bg-card'
                  )}
                >
                  <div className="text-base font-semibold text-foreground">{item.title}</div>
                  {item.summary && (
                    <div className="mt-1 text-sm text-muted-foreground">{item.summary}</div>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Right preview: sticky code panel */}
          <div className="mt-8 lg:col-span-7 lg:mt-0">
            <div className="lg:sticky lg:top-24">
              <div
                role="tabpanel"
                id={`panel-${activeExample.id}`}
                aria-labelledby={`tab-${activeExample.id}`}
                className="grid h-full grid-rows-[auto,1fr] overflow-hidden rounded-xl border border-border bg-card"
              >
                <div key={activeExample.id} className="contents animate-in fade-in-50 duration-200">
                  {/* Chrome bar */}
                  <div className="flex items-center justify-between border-b border-border bg-muted/60 px-4 py-2">
                    <span className="text-sm text-muted-foreground">
                      {activeExample.badge || activeExample.language || 'Example'}
                    </span>
                    <button
                      onClick={handleCopy}
                      className="text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                      aria-label="Copy code"
                    >
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>

                  {/* Code area */}
                  <div className="overflow-auto p-4">
                    <pre className="h-full font-mono text-sm text-foreground whitespace-pre">
                      <code>{activeExample.code}</code>
                    </pre>
                  </div>
                </div>
              </div>

              {/* Optional footer note */}
              {footerNote && (
                <p className="mt-4 text-center text-sm text-muted-foreground">{footerNote}</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Screen reader live region for copy feedback */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {copied && 'Code copied to clipboard'}
      </div>
    </section>
  )
}

// Default props for quick usage
CodeExamplesSection.defaultProps = {
  title: 'Build AI agents with llama-orch-utils',
  subtitle: 'TypeScript utilities for LLM pipelines and agentic workflows.',
  items: [
    {
      id: 'simple',
      title: 'Simple code generation',
      summary: 'Invoke to generate a TypeScript validator.',
      language: 'TypeScript',
      code: `import { invoke } from '@llama-orch/utils';

const response = await invoke({
  prompt: 'Generate a TypeScript function that validates email addresses',
  model: 'llama-3.1-70b',
  maxTokens: 500
});

console.log(response.text);`,
    },
    {
      id: 'files',
      title: 'File operations',
      summary: 'Read schema → generate API → write file.',
      language: 'TypeScript',
      code: `import { FileReader, FileWriter, invoke } from '@llama-orch/utils';

// Read schema
const schema = await FileReader.read('schema.sql');

// Generate API
const code = await invoke({
  prompt: \`Generate TypeScript CRUD API for:\\n\${schema}\`,
  model: 'llama-3.1-70b'
});

// Write result
await FileWriter.write('src/api.ts', code.text);`,
    },
    {
      id: 'agent',
      title: 'Multi-step agent',
      summary: 'Threaded review + suggestion extraction.',
      language: 'TypeScript',
      code: `import { Thread, invoke, extractCode } from '@llama-orch/utils';

// Build conversation thread
const thread = Thread.create()
  .addSystem('You are a code review expert')
  .addUser('Review this code for security issues')
  .addUser(await FileReader.read('src/auth.ts'));

// Get review
const review = await invoke({
  messages: thread.toMessages(),
  model: 'llama-3.1-70b'
});

// Extract suggestions
const suggestions = extractCode(review.text, 'typescript');
await FileWriter.write('review.md', review.text);`,
    },
  ],
}
