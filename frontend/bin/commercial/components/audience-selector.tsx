import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ArrowRight, Code2, Server, Shield } from "lucide-react"
import Link from "next/link"

export function AudienceSelector() {
  return (
    <section className="relative bg-background py-24 sm:py-32">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent" />

      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto mb-16 max-w-3xl text-center">
          <p className="mb-4 font-sans text-sm font-medium uppercase tracking-wider text-primary">Choose Your Path</p>

          <h2 className="mb-6 font-sans text-3xl font-semibold tracking-tight text-foreground sm:text-4xl lg:text-5xl">
            Where do you want to start?
          </h2>
          <p className="font-sans text-lg leading-relaxed text-muted-foreground">
            rbee adapts to your needs. Whether you're building, providing, or securing—we have a path designed for you.
          </p>
        </div>

        <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-3 lg:gap-8">
          {/* Developers Card */}
          <Card className="group relative overflow-hidden border-border bg-card p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-chart-2/50 hover:bg-card/80 hover:shadow-2xl hover:shadow-chart-2/20">
            <div className="absolute inset-0 -z-10 bg-gradient-to-br from-chart-2/0 via-chart-2/0 to-chart-2/0 opacity-0 transition-all duration-500 group-hover:from-chart-2/5 group-hover:via-chart-2/10 group-hover:to-transparent group-hover:opacity-100" />

            <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-chart-2 to-chart-2 shadow-lg shadow-chart-2/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-chart-2/50">
              <Code2 className="h-7 w-7 text-primary-foreground" />
            </div>

            <div className="mb-2 text-sm font-medium uppercase tracking-wider text-chart-2">For Developers</div>
            <h3 className="mb-3 font-sans text-2xl font-semibold text-card-foreground">Build on Your Hardware</h3>

            <p className="mb-6 font-sans leading-relaxed text-muted-foreground">
              Use your homelab GPUs to power AI coding tools. OpenAI-compatible API works with Zed, Cursor, and any tool
              you already use.
            </p>

            <ul className="mb-8 space-y-3 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="mt-1 text-chart-2">→</span>
                <span>Zero API costs, unlimited usage</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-chart-2">→</span>
                <span>Complete privacy, no data leaves your network</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-chart-2">→</span>
                <span>Build custom AI agents with TypeScript</span>
              </li>
            </ul>

            <Link href="/developers" className="block">
              <Button className="w-full bg-chart-2 font-medium text-primary-foreground transition-all hover:bg-chart-2/90 hover:shadow-lg hover:shadow-chart-2/30">
                Explore Developer Path
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </Card>

          {/* GPU Providers Card */}
          <Card className="group relative overflow-hidden border-border bg-card p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-chart-3/50 hover:bg-card/80 hover:shadow-2xl hover:shadow-chart-3/20">
            <div className="absolute inset-0 -z-10 bg-gradient-to-br from-chart-3/0 via-chart-3/0 to-chart-3/0 opacity-0 transition-all duration-500 group-hover:from-chart-3/5 group-hover:via-chart-3/10 group-hover:to-transparent group-hover:opacity-100" />

            <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-chart-3 to-chart-3 shadow-lg shadow-chart-3/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-chart-3/50">
              <Server className="h-7 w-7 text-primary-foreground" />
            </div>

            <div className="mb-2 text-sm font-medium uppercase tracking-wider text-chart-3">For GPU Owners</div>
            <h3 className="mb-3 font-sans text-2xl font-semibold text-card-foreground">Monetize Your Hardware</h3>

            <p className="mb-6 font-sans leading-relaxed text-muted-foreground">
              Turn idle GPUs into revenue. Join the rbee marketplace and earn from your gaming PC, workstation, or
              server farm.
            </p>

            <ul className="mb-8 space-y-3 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="mt-1 text-chart-3">→</span>
                <span>Set your own pricing and availability</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-chart-3">→</span>
                <span>Secure platform with audit trails</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-chart-3">→</span>
                <span>Passive income from existing hardware</span>
              </li>
            </ul>

            <Link href="/gpu-providers" className="block">
              <Button className="w-full bg-chart-3 font-medium text-primary-foreground transition-all hover:bg-chart-3/90 hover:shadow-lg hover:shadow-chart-3/30">
                Become a Provider
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </Card>

          {/* Enterprise Card */}
          <Card className="group relative overflow-hidden border-border bg-card p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-primary/50 hover:bg-card/80 hover:shadow-2xl hover:shadow-primary/20">
            <div className="absolute inset-0 -z-10 bg-gradient-to-br from-primary/0 via-primary/0 to-primary/0 opacity-0 transition-all duration-500 group-hover:from-primary/5 group-hover:via-primary/10 group-hover:to-transparent group-hover:opacity-100" />

            <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary shadow-lg shadow-primary/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-primary/50">
              <Shield className="h-7 w-7 text-primary-foreground" />
            </div>

            <div className="mb-2 text-sm font-medium uppercase tracking-wider text-primary">For Enterprise</div>
            <h3 className="mb-3 font-sans text-2xl font-semibold text-card-foreground">Compliance & Security</h3>

            <p className="mb-6 font-sans leading-relaxed text-muted-foreground">
              Deploy AI infrastructure with EU compliance, comprehensive audit trails, and enterprise-grade security
              built-in from day one.
            </p>

            <ul className="mb-8 space-y-3 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="mt-1 text-primary">→</span>
                <span>GDPR-compliant with 7-year audit retention</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-primary">→</span>
                <span>SOC2 and ISO 27001 aligned</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-primary">→</span>
                <span>On-premises or private cloud deployment</span>
              </li>
            </ul>

            <Link href="/enterprise" className="block">
              <Button className="w-full bg-primary font-medium text-primary-foreground transition-all hover:bg-primary/90 hover:shadow-lg hover:shadow-primary/30">
                Enterprise Solutions
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </Card>
        </div>

        <div className="mx-auto mt-12 max-w-2xl text-center">
          <p className="font-sans text-sm leading-relaxed text-muted-foreground">
            Not sure which path fits? Keep scrolling to explore the full platform.
          </p>
        </div>
      </div>
    </section>
  )
}
