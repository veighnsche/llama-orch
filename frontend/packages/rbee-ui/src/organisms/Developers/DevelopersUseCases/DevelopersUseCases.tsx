import { UseCasesSection } from '@rbee/ui/organisms'
import { Code, FileText, FlaskConical, GitPullRequest, Wrench } from 'lucide-react'

export function DevelopersUseCases() {
  return (
    <UseCasesSection
      id="use-cases"
      title="Built for developers who value independence"
      items={[
        {
          icon: Code,
          title: 'Build your own AI coder',
          scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
          solution: 'Run rbee on a gaming PC + old workstation. Llama-3.1-70B for code; Stable Diffusion for assets.',
          outcome: '$0/month AI costs. Full control. No rate limits.',
          tags: ['OpenAI-compatible', 'Local models'],
        },
        {
          icon: FileText,
          title: 'Documentation generators',
          scenario: 'Need comprehensive docs from codebase; API costs are prohibitive.',
          solution: 'Process entire repos locally with rbee. Generate markdown with examples.',
          outcome: 'Process entire repos. Zero API costs. Private by default.',
          tags: ['Markdown', 'Privacy'],
        },
        {
          icon: FlaskConical,
          title: 'Test generators',
          scenario: 'Writing tests is time-consuming; need AI to generate comprehensive suites.',
          solution: 'Use rbee + llama-orch-utils to generate Jest/Vitest tests from specs.',
          outcome: '10× faster coverage. No external dependencies.',
          tags: ['Jest', 'Vitest'],
        },
        {
          icon: GitPullRequest,
          title: 'Code review agents',
          scenario: 'Small team needs automated code review but cannot afford enterprise tools.',
          solution: 'Build custom review agent with rbee. Analyze PRs for issues, security, performance.',
          outcome: 'Automated reviews. Zero ongoing costs. Custom rules.',
          tags: ['GitHub', 'GitLab'],
        },
        {
          icon: Wrench,
          title: 'Refactoring agents',
          scenario: 'Legacy codebase needs modernization; manual refactoring would take months.',
          solution: 'Use rbee to refactor code to modern patterns. TypeScript, async/await, etc.',
          outcome: 'Months of work → days. You approve every change.',
          tags: ['TypeScript', 'Modernization'],
        },
      ]}
    />
  )
}
