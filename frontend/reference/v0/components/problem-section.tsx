import { AlertTriangle, DollarSign, Lock } from "lucide-react"

export function ProblemSection() {
  return (
    <section className="py-24 bg-gradient-to-br from-red-950/20 via-slate-900 to-slate-900">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6 text-balance">
            The Hidden Cost of AI Dependency
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {/* Card 1 */}
          <div className="bg-slate-900/50 border border-red-900/30 rounded-lg p-8 space-y-4 hover:border-red-700/50 transition-colors">
            <div className="h-12 w-12 rounded-lg bg-red-500/10 flex items-center justify-center">
              <AlertTriangle className="h-6 w-6 text-red-400" />
            </div>
            <h3 className="text-xl font-bold text-white">Loss of Control</h3>
            <p className="text-slate-300 leading-relaxed">
              Providers change models, deprecate APIs, or shut down entirely. Your infrastructure, workflows, and
              business continuity depend on decisions you can't control.
            </p>
          </div>

          {/* Card 2 */}
          <div className="bg-slate-900/50 border border-red-900/30 rounded-lg p-8 space-y-4 hover:border-red-700/50 transition-colors">
            <div className="h-12 w-12 rounded-lg bg-red-500/10 flex items-center justify-center">
              <DollarSign className="h-6 w-6 text-red-400" />
            </div>
            <h3 className="text-xl font-bold text-white">Unpredictable Costs</h3>
            <p className="text-slate-300 leading-relaxed">
              Pricing changes without warning. Usage scales, costs spiral. What starts affordable becomes unsustainable
              as your needs grow.
            </p>
          </div>

          {/* Card 3 */}
          <div className="bg-slate-900/50 border border-red-900/30 rounded-lg p-8 space-y-4 hover:border-red-700/50 transition-colors">
            <div className="h-12 w-12 rounded-lg bg-red-500/10 flex items-center justify-center">
              <Lock className="h-6 w-6 text-red-400" />
            </div>
            <h3 className="text-xl font-bold text-white">Privacy & Compliance Risks</h3>
            <p className="text-slate-300 leading-relaxed">
              Sensitive data leaves your network. Compliance requirements clash with cloud dependencies. Audit trails
              are incomplete. Regulatory exposure grows.
            </p>
          </div>
        </div>

        <div className="max-w-3xl mx-auto text-center mt-12">
          <p className="text-xl text-slate-300 leading-relaxed text-balance">
            Whether you're building with AI, monetizing hardware, or ensuring compliance—dependency on external
            providers creates risk you can't afford.
          </p>
        </div>
      </div>
    </section>
  )
}
