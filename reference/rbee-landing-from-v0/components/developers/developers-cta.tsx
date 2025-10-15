import { Button } from "@/components/ui/button"
import { ArrowRight, Github } from "lucide-react"

export function DevelopersCTA() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-slate-950 via-amber-950/20 to-slate-950 py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Stop Depending on AI Providers. Start Building Today.
          </h2>
          <p className="mb-8 text-balance text-lg leading-relaxed text-slate-300">
            Join 500+ developers who&apos;ve taken control of their AI infrastructure.
          </p>

          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Button size="lg" className="group bg-amber-500 text-slate-950 hover:bg-amber-400">
              Get Started Free
              <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-slate-700 text-white hover:bg-slate-800 bg-transparent"
            >
              <Github className="mr-2 h-4 w-4" />
              View Documentation
            </Button>
          </div>

          <p className="mt-8 text-sm text-slate-400">
            100% open source. No credit card required. Install in 15 minutes.
          </p>
        </div>
      </div>
    </section>
  )
}
