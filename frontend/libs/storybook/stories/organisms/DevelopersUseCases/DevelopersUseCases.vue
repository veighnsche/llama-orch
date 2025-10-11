<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersUseCases -->
<script setup lang="ts">
import { Code, FileText, FlaskConical, GitPullRequest, Wrench } from 'lucide-vue-next'

interface UseCase {
  icon: 'code' | 'file-text' | 'flask' | 'git-pr' | 'wrench'
  title: string
  scenario: string
  solution: string
  outcome: string
}

interface Props {
  title?: string
  useCases?: UseCase[]
}

withDefaults(defineProps<Props>(), {
  title: 'Built for Developers Who Value Independence',
  useCases: () => [
    {
      icon: 'code',
      title: 'Build Your Own AI Coder',
      scenario: 'Building a SaaS with AI features. Uses Claude for coding but fears vendor lock-in.',
      solution: 'Runs rbee on gaming PC + old workstation. Llama 70B for coding, Stable Diffusion for assets.',
      outcome: '$0/month AI costs. Complete control. Never blocked by rate limits.',
    },
    {
      icon: 'file-text',
      title: 'Documentation Generators',
      scenario: 'Need to generate comprehensive docs from codebase but API costs are prohibitive.',
      solution: 'Uses rbee to process entire codebase locally. Generates markdown docs with examples.',
      outcome: 'Process unlimited code. Zero API costs. Complete privacy.',
    },
    {
      icon: 'flask',
      title: 'Test Generators',
      scenario: 'Writing tests is time-consuming. Need AI to generate comprehensive test suites.',
      solution: 'Uses rbee + llama-orch-utils to generate Jest/Vitest tests from specifications.',
      outcome: '10x faster test coverage. No external dependencies.',
    },
    {
      icon: 'git-pr',
      title: 'Code Review Agents',
      scenario: "Small team needs automated code review but can't afford enterprise tools.",
      solution: 'Builds custom review agent with rbee. Analyzes PRs for issues, security, performance.',
      outcome: 'Automated reviews. Zero ongoing costs. Custom rules.',
    },
    {
      icon: 'wrench',
      title: 'Refactoring Agents',
      scenario: 'Legacy codebase needs modernization. Manual refactoring would take months.',
      solution: 'Uses rbee to refactor code to modern patterns. TypeScript, async/await, etc.',
      outcome: 'Months of work in days. Complete control over changes.',
    },
  ],
})

const getIcon = (iconType: string) => {
  switch (iconType) {
    case 'code':
      return Code
    case 'file-text':
      return FileText
    case 'flask':
      return FlaskConical
    case 'git-pr':
      return GitPullRequest
    case 'wrench':
      return Wrench
    default:
      return Code
  }
}
</script>

<template>
  <section class="border-b border-border bg-secondary/30 py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
      </div>

      <div class="mx-auto mt-16 grid max-w-6xl gap-8 sm:grid-cols-2 lg:grid-cols-3">
        <div
          v-for="(useCase, index) in useCases"
          :key="index"
          class="group rounded-lg border border-border bg-secondary/50 p-6 transition-all hover:border-primary/50 hover:bg-secondary"
        >
          <div class="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
            <component :is="getIcon(useCase.icon)" class="h-6 w-6 text-primary" />
          </div>
          <h3 class="mb-3 text-lg font-semibold text-foreground">{{ useCase.title }}</h3>
          <div class="space-y-3 text-sm">
            <div>
              <div class="mb-1 font-medium text-muted-foreground">Scenario</div>
              <div class="text-balance leading-relaxed text-foreground">{{ useCase.scenario }}</div>
            </div>
            <div>
              <div class="mb-1 font-medium text-muted-foreground">Solution</div>
              <div class="text-balance leading-relaxed text-foreground">{{ useCase.solution }}</div>
            </div>
            <div>
              <div class="mb-1 font-medium text-primary">Outcome</div>
              <div class="text-balance leading-relaxed text-foreground">{{ useCase.outcome }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
