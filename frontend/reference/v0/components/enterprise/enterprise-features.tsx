import { Shield, Users, Wrench, Globe } from "lucide-react"

export function EnterpriseFeatures() {
  return (
    <section className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-white">Enterprise Features</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-slate-300">
            Everything you need for enterprise-grade AI infrastructure deployment and management.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          {/* Feature 1 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Shield className="h-6 w-6 text-amber-400" />
              </div>
              <h3 className="text-xl font-bold text-white">Enterprise SLAs</h3>
            </div>
            <p className="mb-4 leading-relaxed text-slate-300">
              99.9% uptime guarantee with 24/7 support and 1-hour response time. Dedicated account manager and quarterly
              business reviews.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>99.9% uptime SLA</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>24/7 support (1-hour response)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Dedicated account manager</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Quarterly business reviews</span>
              </li>
            </ul>
          </div>

          {/* Feature 2 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Users className="h-6 w-6 text-amber-400" />
              </div>
              <h3 className="text-xl font-bold text-white">White-Label Option</h3>
            </div>
            <p className="mb-4 leading-relaxed text-slate-300">
              Rebrand rbee as your own AI infrastructure platform. Custom branding, domain, and UI customization
              available.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Custom branding and logo</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Custom domain (ai.yourcompany.com)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>UI customization</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>API endpoint customization</span>
              </li>
            </ul>
          </div>

          {/* Feature 3 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Wrench className="h-6 w-6 text-amber-400" />
              </div>
              <h3 className="text-xl font-bold text-white">Professional Services</h3>
            </div>
            <p className="mb-4 leading-relaxed text-slate-300">
              Expert consulting for deployment, integration, and optimization. Custom development and training
              available.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Deployment consulting</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Integration support</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Custom development</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Team training</span>
              </li>
            </ul>
          </div>

          {/* Feature 4 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Globe className="h-6 w-6 text-amber-400" />
              </div>
              <h3 className="text-xl font-bold text-white">Multi-Region Support</h3>
            </div>
            <p className="mb-4 leading-relaxed text-slate-300">
              Deploy across multiple EU regions for redundancy and compliance. Automatic failover and load balancing.
            </p>
            <ul className="space-y-2 text-sm text-slate-400">
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Multi-region deployment (EU)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Automatic failover</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Load balancing</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-400">✓</span>
                <span>Geo-redundancy</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
