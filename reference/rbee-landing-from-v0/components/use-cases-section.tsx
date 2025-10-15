import { Building, Home, Laptop, Users } from "lucide-react"

export function UseCasesSection() {
  return (
    <section className="py-24 bg-white">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
            Built for Those Who Value Independence
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Use Case 1 */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center">
              <Laptop className="h-6 w-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">The Solo Developer</h3>
            <div className="space-y-3 text-sm">
              <p className="text-slate-600">
                <span className="font-medium text-slate-900">Scenario:</span> Building a SaaS with AI features. Uses
                Claude for coding but fears vendor lock-in.
              </p>
              <p className="text-slate-600">
                <span className="font-medium text-slate-900">Solution:</span> Runs rbee on gaming PC + old workstation.
                Llama 70B for coding, Stable Diffusion for assets.
              </p>
              <p className="text-green-700 font-medium">
                ✓ $0/month AI costs. Complete control. Never blocked by rate limits.
              </p>
            </div>
          </div>

          {/* Use Case 2 */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-amber-100 flex items-center justify-center">
              <Users className="h-6 w-6 text-amber-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">The Small Team</h3>
            <div className="space-y-3 text-sm">
              <p className="text-slate-600">
                <span className="font-medium text-slate-900">Scenario:</span> 5-person startup. Spending $500/month on
                AI APIs. Need to cut costs.
              </p>
              <p className="text-slate-600">
                <span className="font-medium text-slate-900">Solution:</span> Pools team's hardware. 3 workstations + 2
                Macs = 8 GPUs total. Shared rbee cluster.
              </p>
              <p className="text-green-700 font-medium">✓ Saves $6,000/year. Faster inference. GDPR-compliant.</p>
            </div>
          </div>

          {/* Use Case 3 */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center">
              <Home className="h-6 w-6 text-green-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">The Homelab Enthusiast</h3>
            <div className="space-y-3 text-sm">
              <p className="text-slate-600">
                <span className="font-medium text-slate-900">Scenario:</span> Has 4 GPUs collecting dust. Wants to build
                AI agents for personal projects.
              </p>
              <p className="text-slate-600">
                <span className="font-medium text-slate-900">Solution:</span> Runs rbee across homelab. Builds custom AI
                coder, documentation generator, code reviewer.
              </p>
              <p className="text-green-700 font-medium">✓ Turns idle hardware into productive AI infrastructure.</p>
            </div>
          </div>

          {/* Use Case 4 */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 space-y-4">
            <div className="h-12 w-12 rounded-lg bg-slate-100 flex items-center justify-center">
              <Building className="h-6 w-6 text-slate-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900">The Enterprise</h3>
            <div className="space-y-3 text-sm">
              <p className="text-slate-600">
                <span className="font-medium text-slate-900">Scenario:</span> 50-person dev team. Can't send code to
                external APIs due to compliance.
              </p>
              <p className="text-slate-600">
                <span className="font-medium text-slate-900">Solution:</span> Deploys rbee on-premises. 20 GPUs across
                data center. Custom Rhai routing for compliance.
              </p>
              <p className="text-green-700 font-medium">
                ✓ EU-only routing. Full audit trail. Zero external dependencies.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
