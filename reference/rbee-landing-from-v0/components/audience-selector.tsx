import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ArrowRight, Code2, Server, Shield } from "lucide-react"
import Link from "next/link"

export function AudienceSelector() {
  return (
    <section className="relative bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 py-24 sm:py-32">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-amber-500/20 to-transparent" />

      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto mb-16 max-w-3xl text-center">
          <p className="mb-4 font-sans text-sm font-medium uppercase tracking-wider text-amber-500">Choose Your Path</p>

          <h2 className="mb-6 font-sans text-3xl font-semibold tracking-tight text-white sm:text-4xl lg:text-5xl">
            Where do you want to start?
          </h2>
          <p className="font-sans text-lg leading-relaxed text-slate-400">
            rbee adapts to your needs. Whether you're building, providing, or securing—we have a path designed for you.
          </p>
        </div>

        <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-3 lg:gap-8">
          {/* Developers Card */}
          <Card className="group relative overflow-hidden border-slate-800/50 bg-slate-900/30 p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-blue-500/50 hover:bg-slate-900/60 hover:shadow-2xl hover:shadow-blue-500/20">
            <div className="absolute inset-0 -z-10 bg-gradient-to-br from-blue-500/0 via-blue-500/0 to-blue-500/0 opacity-0 transition-all duration-500 group-hover:from-blue-500/5 group-hover:via-blue-500/10 group-hover:to-transparent group-hover:opacity-100" />

            <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 shadow-lg shadow-blue-500/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-blue-500/50">
              <Code2 className="h-7 w-7 text-white" />
            </div>

            <div className="mb-2 text-sm font-medium uppercase tracking-wider text-blue-400">For Developers</div>
            <h3 className="mb-3 font-sans text-2xl font-semibold text-white">Build on Your Hardware</h3>

            <p className="mb-6 font-sans leading-relaxed text-slate-400">
              Use your homelab GPUs to power AI coding tools. OpenAI-compatible API works with Zed, Cursor, and any tool
              you already use.
            </p>

            <ul className="mb-8 space-y-3 text-sm text-slate-300">
              <li className="flex items-start gap-2">
                <span className="mt-1 text-blue-400">→</span>
                <span>Zero API costs, unlimited usage</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-blue-400">→</span>
                <span>Complete privacy, no data leaves your network</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-blue-400">→</span>
                <span>Build custom AI agents with TypeScript</span>
              </li>
            </ul>

            <Link href="/developers" className="block">
              <Button className="w-full bg-blue-600 font-medium text-white transition-all hover:bg-blue-700 hover:shadow-lg hover:shadow-blue-500/30">
                Explore Developer Path
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </Card>

          {/* GPU Providers Card */}
          <Card className="group relative overflow-hidden border-slate-800/50 bg-slate-900/30 p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-green-500/50 hover:bg-slate-900/60 hover:shadow-2xl hover:shadow-green-500/20">
            <div className="absolute inset-0 -z-10 bg-gradient-to-br from-green-500/0 via-green-500/0 to-green-500/0 opacity-0 transition-all duration-500 group-hover:from-green-500/5 group-hover:via-green-500/10 group-hover:to-transparent group-hover:opacity-100" />

            <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-green-500 to-green-600 shadow-lg shadow-green-500/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-green-500/50">
              <Server className="h-7 w-7 text-white" />
            </div>

            <div className="mb-2 text-sm font-medium uppercase tracking-wider text-green-400">For GPU Owners</div>
            <h3 className="mb-3 font-sans text-2xl font-semibold text-white">Monetize Your Hardware</h3>

            <p className="mb-6 font-sans leading-relaxed text-slate-400">
              Turn idle GPUs into revenue. Join the rbee marketplace and earn from your gaming PC, workstation, or
              server farm.
            </p>

            <ul className="mb-8 space-y-3 text-sm text-slate-300">
              <li className="flex items-start gap-2">
                <span className="mt-1 text-green-400">→</span>
                <span>Set your own pricing and availability</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-green-400">→</span>
                <span>Secure platform with audit trails</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-green-400">→</span>
                <span>Passive income from existing hardware</span>
              </li>
            </ul>

            <Link href="/gpu-providers" className="block">
              <Button className="w-full bg-green-600 font-medium text-white transition-all hover:bg-green-700 hover:shadow-lg hover:shadow-green-500/30">
                Become a Provider
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </Card>

          {/* Enterprise Card */}
          <Card className="group relative overflow-hidden border-slate-800/50 bg-slate-900/30 p-8 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:border-amber-500/50 hover:bg-slate-900/60 hover:shadow-2xl hover:shadow-amber-500/20">
            <div className="absolute inset-0 -z-10 bg-gradient-to-br from-amber-500/0 via-amber-500/0 to-amber-500/0 opacity-0 transition-all duration-500 group-hover:from-amber-500/5 group-hover:via-amber-500/10 group-hover:to-transparent group-hover:opacity-100" />

            <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 shadow-lg shadow-amber-500/30 transition-all duration-300 group-hover:scale-110 group-hover:shadow-amber-500/50">
              <Shield className="h-7 w-7 text-white" />
            </div>

            <div className="mb-2 text-sm font-medium uppercase tracking-wider text-amber-400">For Enterprise</div>
            <h3 className="mb-3 font-sans text-2xl font-semibold text-white">Compliance & Security</h3>

            <p className="mb-6 font-sans leading-relaxed text-slate-400">
              Deploy AI infrastructure with EU compliance, comprehensive audit trails, and enterprise-grade security
              built-in from day one.
            </p>

            <ul className="mb-8 space-y-3 text-sm text-slate-300">
              <li className="flex items-start gap-2">
                <span className="mt-1 text-amber-400">→</span>
                <span>GDPR-compliant with 7-year audit retention</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-amber-400">→</span>
                <span>SOC2 and ISO 27001 aligned</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="mt-1 text-amber-400">→</span>
                <span>On-premises or private cloud deployment</span>
              </li>
            </ul>

            <Link href="/enterprise" className="block">
              <Button className="w-full bg-amber-600 font-medium text-white transition-all hover:bg-amber-700 hover:shadow-lg hover:shadow-amber-500/30">
                Enterprise Solutions
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
          </Card>
        </div>

        <div className="mx-auto mt-12 max-w-2xl text-center">
          <p className="font-sans text-sm leading-relaxed text-slate-500">
            Not sure which path fits? Keep scrolling to explore the full platform.
          </p>
        </div>
      </div>
    </section>
  )
}
