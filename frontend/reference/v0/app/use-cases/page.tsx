import { Building, Home, Laptop, Users, Briefcase, GraduationCap, Code, Server } from "lucide-react"
import { EmailCapture } from "@/components/email-capture"

export default function UseCasesPage() {
  return (
    <div className="pt-16">
      {/* Hero Section */}
      <section className="py-24 bg-gradient-to-b from-slate-950 to-slate-900">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl lg:text-6xl font-bold text-white mb-6 text-balance">
              Built for Those Who Value
              <br />
              <span className="text-amber-500">Independence</span>
            </h1>
            <p className="text-xl text-slate-300 leading-relaxed">
              From solo developers to enterprises, rbee adapts to your needs. Own your AI infrastructure without
              compromising on power or flexibility.
            </p>
          </div>
        </div>
      </section>

      {/* Primary Use Cases */}
      <section className="py-24 bg-white">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
              Real Scenarios. Real Solutions.
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
                  <span className="font-medium text-slate-900">Solution:</span> Runs rbee on gaming PC + old
                  workstation. Llama 70B for coding, Stable Diffusion for assets.
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
                  <span className="font-medium text-slate-900">Solution:</span> Pools team's hardware. 3 workstations +
                  2 Macs = 8 GPUs total. Shared rbee cluster.
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
                  <span className="font-medium text-slate-900">Scenario:</span> Has 4 GPUs collecting dust. Wants to
                  build AI agents for personal projects.
                </p>
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Solution:</span> Runs rbee across homelab. Builds custom
                  AI coder, documentation generator, code reviewer.
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

            {/* Use Case 5 */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 space-y-4">
              <div className="h-12 w-12 rounded-lg bg-amber-100 flex items-center justify-center">
                <Briefcase className="h-6 w-6 text-amber-600" />
              </div>
              <h3 className="text-xl font-bold text-slate-900">The Freelance Developer</h3>
              <div className="space-y-3 text-sm">
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Scenario:</span> Works on multiple client projects. Needs
                  AI assistance but can't share client code with external APIs.
                </p>
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Solution:</span> Runs rbee locally. All client code stays
                  on local machine. Uses Llama for code generation and review.
                </p>
                <p className="text-green-700 font-medium">
                  ✓ Client confidentiality maintained. Professional AI tools. Zero subscription costs.
                </p>
              </div>
            </div>

            {/* Use Case 6 */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 space-y-4">
              <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center">
                <GraduationCap className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="text-xl font-bold text-slate-900">The Research Lab</h3>
              <div className="space-y-3 text-sm">
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Scenario:</span> University research lab with grant
                  funding. Needs AI for research but limited budget for cloud services.
                </p>
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Solution:</span> Deploys rbee on lab's GPU cluster. Uses
                  grant money for hardware, not subscriptions. Full control over models.
                </p>
                <p className="text-green-700 font-medium">
                  ✓ Maximizes research budget. Reproducible experiments. No vendor dependency.
                </p>
              </div>
            </div>

            {/* Use Case 7 */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 space-y-4">
              <div className="h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center">
                <Code className="h-6 w-6 text-green-600" />
              </div>
              <h3 className="text-xl font-bold text-slate-900">The Open Source Maintainer</h3>
              <div className="space-y-3 text-sm">
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Scenario:</span> Maintains popular open source projects.
                  Wants AI assistance for code reviews and documentation but can't afford enterprise AI tools.
                </p>
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Solution:</span> Runs rbee on personal hardware. Builds
                  custom AI agents for PR reviews, documentation generation, and issue triage.
                </p>
                <p className="text-green-700 font-medium">
                  ✓ Sustainable AI tooling. Community-aligned. Zero ongoing costs.
                </p>
              </div>
            </div>

            {/* Use Case 8 */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 space-y-4">
              <div className="h-12 w-12 rounded-lg bg-slate-100 flex items-center justify-center">
                <Server className="h-6 w-6 text-slate-600" />
              </div>
              <h3 className="text-xl font-bold text-slate-900">The GPU Provider</h3>
              <div className="space-y-3 text-sm">
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Scenario:</span> Has idle GPU hardware (former mining
                  rig, gaming PC). Wants to monetize excess capacity.
                </p>
                <p className="text-slate-600">
                  <span className="font-medium text-slate-900">Solution:</span> Joins rbee marketplace (M3). Sets
                  pricing and availability. Earns passive income from idle hardware.
                </p>
                <p className="text-green-700 font-medium">
                  ✓ Passive income stream. Help the community. Control when GPUs are available.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Industry-Specific Use Cases */}
      <section className="py-24 bg-slate-50">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
              Industry-Specific Solutions
            </h2>
            <p className="text-xl text-slate-600 leading-relaxed">
              rbee adapts to the unique compliance and security requirements of regulated industries.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-bold text-slate-900">Financial Services</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                GDPR compliance, audit trails, data residency controls. AI-powered code review and risk analysis without
                sending sensitive financial data to external APIs.
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-bold text-slate-900">Healthcare</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                HIPAA-compliant infrastructure. Patient data never leaves your network. AI-assisted medical coding,
                documentation, and research without privacy concerns.
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-bold text-slate-900">Legal</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                Attorney-client privilege maintained. Document analysis, contract review, and legal research with AI
                while keeping all client information confidential.
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-bold text-slate-900">Government</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                Sovereign AI infrastructure. No foreign cloud dependencies. Complete audit trails for compliance with
                government security standards.
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-bold text-slate-900">Education</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                Student data protection. AI-powered tutoring, grading assistance, and research tools without sending
                student information to external services.
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-bold text-slate-900">Manufacturing</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                Protect trade secrets and proprietary designs. AI-assisted CAD, quality control, and process
                optimization without exposing intellectual property.
              </p>
            </div>
          </div>
        </div>
      </section>

      <EmailCapture />
    </div>
  )
}
