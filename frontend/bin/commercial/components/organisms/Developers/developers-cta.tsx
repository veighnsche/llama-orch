import { Button } from '@/components/atoms/Button/Button'
import { ArrowRight, Github } from "lucide-react"

export function DevelopersCTA() {
  return (
    <section className="border-b border-border bg-background py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Stop Depending on AI Providers. Start Building Today.
          </h2>
          <p className="mb-8 text-balance text-lg leading-relaxed text-muted-foreground">
            Join 500+ developers who&apos;ve taken control of their AI infrastructure.
          </p>

          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <Button size="lg" className="group bg-primary text-primary-foreground hover:bg-primary/90">
              Get Started Free
              <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-border text-foreground hover:bg-secondary bg-transparent"
            >
              <Github className="mr-2 h-4 w-4" />
              View Documentation
            </Button>
          </div>

          <p className="mt-8 text-sm text-muted-foreground">
            100% open source. No credit card required. Install in 15 minutes.
          </p>
        </div>
      </div>
    </section>
  )
}
