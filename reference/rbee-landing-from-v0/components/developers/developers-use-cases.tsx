import { Code, FileText, FlaskConical, GitPullRequest, Wrench } from "lucide-react"

const useCases = [
  {
    icon: Code,
    title: "Build Your Own AI Coder",
    scenario: "Building a SaaS with AI features. Uses Claude for coding but fears vendor lock-in.",
    solution: "Runs rbee on gaming PC + old workstation. Llama 70B for coding, Stable Diffusion for assets.",
    outcome: "$0/month AI costs. Complete control. Never blocked by rate limits.",
  },
  {
    icon: FileText,
    title: "Documentation Generators",
    scenario: "Need to generate comprehensive docs from codebase but API costs are prohibitive.",
    solution: "Uses rbee to process entire codebase locally. Generates markdown docs with examples.",
    outcome: "Process unlimited code. Zero API costs. Complete privacy.",
  },
  {
    icon: FlaskConical,
    title: "Test Generators",
    scenario: "Writing tests is time-consuming. Need AI to generate comprehensive test suites.",
    solution: "Uses rbee + llama-orch-utils to generate Jest/Vitest tests from specifications.",
    outcome: "10x faster test coverage. No external dependencies.",
  },
  {
    icon: GitPullRequest,
    title: "Code Review Agents",
    scenario: "Small team needs automated code review but can&apos;t afford enterprise tools.",
    solution: "Builds custom review agent with rbee. Analyzes PRs for issues, security, performance.",
    outcome: "Automated reviews. Zero ongoing costs. Custom rules.",
  },
  {
    icon: Wrench,
    title: "Refactoring Agents",
    scenario: "Legacy codebase needs modernization. Manual refactoring would take months.",
    solution: "Uses rbee to refactor code to modern patterns. TypeScript, async/await, etc.",
    outcome: "Months of work in days. Complete control over changes.",
  },
]

export function DevelopersUseCases() {
  return (
    <section className="border-b border-slate-800 bg-slate-900/30 py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Built for Developers Who Value Independence
          </h2>
        </div>

        <div className="mx-auto mt-16 grid max-w-6xl gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {useCases.map((useCase, index) => (
            <div
              key={index}
              className="group rounded-lg border border-slate-800 bg-slate-900/50 p-6 transition-all hover:border-amber-500/50 hover:bg-slate-900"
            >
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <useCase.icon className="h-6 w-6 text-amber-400" />
              </div>
              <h3 className="mb-3 text-lg font-semibold text-white">{useCase.title}</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <div className="mb-1 font-medium text-slate-400">Scenario</div>
                  <div className="text-balance leading-relaxed text-slate-300">{useCase.scenario}</div>
                </div>
                <div>
                  <div className="mb-1 font-medium text-slate-400">Solution</div>
                  <div className="text-balance leading-relaxed text-slate-300">{useCase.solution}</div>
                </div>
                <div>
                  <div className="mb-1 font-medium text-amber-400">Outcome</div>
                  <div className="text-balance leading-relaxed text-white">{useCase.outcome}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
