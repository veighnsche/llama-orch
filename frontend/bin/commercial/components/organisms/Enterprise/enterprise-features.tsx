import { Shield, Users, Wrench, Globe } from 'lucide-react'

export function EnterpriseFeatures() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-foreground">Enterprise Features</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            Everything you need for enterprise-grade AI infrastructure deployment and management.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          {/* Feature 1 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Shield className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold text-foreground">Enterprise SLAs</h3>
            </div>
            <p className="mb-4 leading-relaxed text-muted-foreground">
              99.9% uptime guarantee with 24/7 support and 1-hour response time. Dedicated account manager and quarterly
              business reviews.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>99.9% uptime SLA</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>24/7 support (1-hour response)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Dedicated account manager</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Quarterly business reviews</span>
              </li>
            </ul>
          </div>

          {/* Feature 2 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Users className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold text-foreground">White-Label Option</h3>
            </div>
            <p className="mb-4 leading-relaxed text-muted-foreground">
              Rebrand rbee as your own AI infrastructure platform. Custom branding, domain, and UI customization
              available.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Custom branding and logo</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Custom domain (ai.yourcompany.com)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>UI customization</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>API endpoint customization</span>
              </li>
            </ul>
          </div>

          {/* Feature 3 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Wrench className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold text-foreground">Professional Services</h3>
            </div>
            <p className="mb-4 leading-relaxed text-muted-foreground">
              Expert consulting for deployment, integration, and optimization. Custom development and training
              available.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Deployment consulting</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Integration support</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Custom development</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Team training</span>
              </li>
            </ul>
          </div>

          {/* Feature 4 */}
          <div className="rounded-lg border border-border bg-card p-8">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <Globe className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold text-foreground">Multi-Region Support</h3>
            </div>
            <p className="mb-4 leading-relaxed text-muted-foreground">
              Deploy across multiple EU regions for redundancy and compliance. Automatic failover and load balancing.
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Multi-region deployment (EU)</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Automatic failover</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Load balancing</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-chart-3">✓</span>
                <span>Geo-redundancy</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
