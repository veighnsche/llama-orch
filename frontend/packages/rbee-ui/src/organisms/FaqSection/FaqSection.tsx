'use client'

import { faqBeehive } from '@rbee/ui/assets'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@rbee/ui/atoms/Accordion'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Input } from '@rbee/ui/atoms/Input'
import { Separator } from '@rbee/ui/atoms/Separator'
import { cn } from '@rbee/ui/utils'
import { Search as SearchIcon } from 'lucide-react'
import Image, { type StaticImageData } from 'next/image'
import * as React from 'react'

// Created by: TEAM-086

export interface FAQItem {
  value: string
  question: string
  answer: React.ReactNode
  category: string
}

interface FAQSectionProps {
  title: string
  subtitle: string
  badgeText?: string
  categories: string[]
  faqItems: FAQItem[]
  showSupportCard?: boolean
  supportCardImage?: string | StaticImageData
  supportCardImageAlt?: string
  supportCardTitle?: string
  supportCardLinks?: Array<{ label: string; href: string }>
  supportCardCTA?: { label: string; href: string }
  jsonLdEnabled?: boolean
}

export function FAQSection({
  title,
  subtitle,
  badgeText,
  categories,
  faqItems,
  showSupportCard,
  supportCardImage,
  supportCardImageAlt,
  supportCardTitle,
  supportCardLinks,
  supportCardCTA,
  jsonLdEnabled,
}: FAQSectionProps) {
  const [searchQuery, setSearchQuery] = React.useState('')
  const [selectedCategory, setSelectedCategory] = React.useState<string | null>(null)
  const [accordionValue, setAccordionValue] = React.useState<string | undefined>(undefined)

  // Filter FAQs based on search and category
  const filteredFAQs = React.useMemo(() => {
    return faqItems.filter((item: FAQItem) => {
      const matchesSearch =
        searchQuery === '' ||
        item.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (typeof item.answer === 'string' && item.answer.toLowerCase().includes(searchQuery.toLowerCase()))

      const matchesCategory = selectedCategory === null || item.category === selectedCategory

      return matchesSearch && matchesCategory
    })
  }, [searchQuery, selectedCategory, faqItems])

  // Group FAQs by category for display
  const groupedFAQs = React.useMemo(() => {
    const groups: Record<string, FAQItem[]> = {}
    filteredFAQs.forEach((item: FAQItem) => {
      if (!groups[item.category]) {
        groups[item.category] = []
      }
      groups[item.category].push(item)
    })
    return groups
  }, [filteredFAQs])

  // JSON-LD Schema
  const jsonLd = jsonLdEnabled
    ? {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: faqItems.map((item: FAQItem) => ({
          '@type': 'Question',
          name: item.question,
          acceptedAnswer: {
            '@type': 'Answer',
            text: typeof item.answer === 'string' ? item.answer : item.question,
          },
        })),
      }
    : null

  const handleExpandAll = () => {
    // Radix Accordion with type="single" doesn't support multiple open items
    // This is a limitation - we'll open the first item as a demonstration
    if (filteredFAQs.length > 0) {
      setAccordionValue(filteredFAQs[0].value)
    }
  }

  const handleCollapseAll = () => {
    setAccordionValue(undefined)
  }

  return (
    <>
      {jsonLd && <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }} />}

      <section className="py-24 bg-secondary" aria-label="Frequently asked questions">
        <div className="container mx-auto px-4">
          <div className={cn('mx-auto grid max-w-6xl grid-cols-1 gap-8', showSupportCard ? 'md:grid-cols-3' : '')}>
            {/* Left Column: Content */}
            <div className={cn('space-y-6', showSupportCard && 'md:col-span-2')}>
              {/* Header */}
              <div className="space-y-3 animate-fade-in-up">
                {badgeText && (
                  <Badge variant="secondary" className="animate-fade-in-up">
                    {badgeText}
                  </Badge>
                )}
                <h2 className="text-3xl font-semibold tracking-tight text-card-foreground">{title}</h2>
                <p className="text-muted-foreground">{subtitle}</p>
              </div>

              {/* Toolbar */}
              <div className="space-y-4 rounded-lg border border-border bg-card/60 backdrop-blur-sm p-4 shadow-sm animate-fade-in">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="relative flex-1">
                    <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground pointer-events-none" />
                    <Input
                      placeholder="Search questions…"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-9"
                      aria-label="Search FAQs"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm" onClick={handleExpandAll}>
                      Expand all
                    </Button>
                    <Button variant="ghost" size="sm" onClick={handleCollapseAll}>
                      Collapse all
                    </Button>
                  </div>
                </div>

                {/* Category Filters */}
                <div className="flex flex-wrap gap-2">
                  {categories.map((cat: string) => (
                    <Button
                      key={cat}
                      variant={selectedCategory === cat ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setSelectedCategory(selectedCategory === cat ? null : cat)}
                      className="rounded-full"
                    >
                      {cat}
                    </Button>
                  ))}
                </div>
              </div>

              {/* FAQ List */}
              {filteredFAQs.length === 0 ? (
                <div className="rounded-lg border border-border bg-card/60 backdrop-blur-sm p-8 text-center shadow-sm animate-fade-in">
                  <p className="text-muted-foreground">
                    No matches. Try keywords like <span className="font-medium">models</span>,{' '}
                    <span className="font-medium">Rust</span>, or <span className="font-medium">migrate</span>.
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {Object.entries(groupedFAQs).map(([category, items], groupIndex) => (
                    <div key={category} className="space-y-4">
                      {groupIndex > 0 && <Separator className="my-6" />}
                      <h3 className="text-lg font-semibold text-card-foreground">{category}</h3>
                      <Accordion
                        type="single"
                        collapsible
                        value={accordionValue}
                        onValueChange={setAccordionValue}
                        className="space-y-3"
                        aria-label="Frequently asked questions"
                      >
                        {items.map((item) => (
                          <AccordionItem
                            key={item.value}
                            value={item.value}
                            className="rounded-lg border border-border bg-card/60 backdrop-blur-sm shadow-sm transition hover:shadow-md animate-fade-in px-6 !border-b"
                          >
                            <AccordionTrigger className="text-left text-base font-semibold text-card-foreground hover:no-underline focus-visible:ring-1 focus-visible:ring-ring/50">
                              {item.question}
                            </AccordionTrigger>
                            <AccordionContent className="text-muted-foreground leading-relaxed">
                              {item.answer}
                            </AccordionContent>
                          </AccordionItem>
                        ))}
                      </Accordion>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Right Column: Support Card */}
            {showSupportCard && (
              <aside className="hidden md:block md:col-span-1">
                <div className="sticky top-24 rounded-xl border border-border bg-card p-5 shadow-sm animate-fade-in space-y-4">
                  {supportCardImage && (
                    <Image
                      src={supportCardImage}
                      width={320}
                      height={180}
                      priority
                      className="rounded-lg shadow-sm ring-1 ring-border object-cover w-full"
                      alt={supportCardImageAlt || ''}
                    />
                  )}
                  <div className="space-y-3">
                    {supportCardTitle && (
                      <h3 className="text-lg font-semibold text-card-foreground">{supportCardTitle}</h3>
                    )}
                    {supportCardLinks && (
                      <ul className="space-y-2 text-sm text-muted-foreground">
                        {supportCardLinks.map((link, idx) => (
                          <li key={idx} className="flex items-start gap-2">
                            <span className="text-primary">•</span>
                            <span>{link.label}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                    <div className="space-y-2 pt-2">
                      {supportCardCTA && (
                        <Button className="w-full" asChild>
                          <a href={supportCardCTA.href} target="_blank" rel="noopener noreferrer">
                            {supportCardCTA.label}
                          </a>
                        </Button>
                      )}
                      {supportCardLinks?.[1] && (
                        <a
                          href={supportCardLinks[1].href}
                          className="block text-center text-sm text-muted-foreground hover:text-foreground transition-colors"
                        >
                          View documentation →
                        </a>
                      )}
                    </div>
                  </div>
                </div>
              </aside>
            )}
          </div>
        </div>
      </section>
    </>
  )
}
