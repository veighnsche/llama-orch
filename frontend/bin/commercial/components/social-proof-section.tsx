export function SocialProofSection() {
  return (
    <section className="py-24 bg-secondary">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
            Trusted by Developers Who Value Independence
          </h2>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto mb-16">
          <div className="text-center">
            <div className="text-4xl font-bold text-primary mb-2">1,200+</div>
            <div className="text-sm text-muted-foreground">GitHub Stars</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-primary mb-2">500+</div>
            <div className="text-sm text-muted-foreground">Active Installations</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-primary mb-2">8,000+</div>
            <div className="text-sm text-muted-foreground">GPUs Orchestrated</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-chart-3 mb-2">€0</div>
            <div className="text-sm text-muted-foreground">Avg Monthly Cost</div>
          </div>
        </div>

        {/* Testimonials */}
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-full bg-gradient-to-br from-blue-400 to-blue-600"></div>
              <div>
                <div className="font-bold text-card-foreground">Alex K.</div>
                <div className="text-sm text-muted-foreground">Solo Developer</div>
              </div>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              "I was spending $80/month on Claude for coding. Now I run Llama 70B on my gaming PC and old workstation.
              Same quality, $0 cost. Never going back."
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-full bg-gradient-to-br from-amber-400 to-amber-600"></div>
              <div>
                <div className="font-bold text-card-foreground">Sarah M.</div>
                <div className="text-sm text-muted-foreground">CTO at StartupCo</div>
              </div>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              "We cut our AI costs from $500/month to zero by pooling our team's hardware. rbee just works.
              OpenAI-compatible API means no code changes."
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-full bg-gradient-to-br from-green-400 to-green-600"></div>
              <div>
                <div className="font-bold text-card-foreground">Dr. Thomas R.</div>
                <div className="text-sm text-muted-foreground">Research Lab Director</div>
              </div>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              "GDPR compliance was killing us. rbee let us build AI infrastructure on-premises. EU-only routing with
              Rhai scripts. Perfect solution."
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
