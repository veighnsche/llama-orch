export function SocialProofSection() {
  return (
    <section className="py-24 bg-slate-50">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
            Trusted by Developers Who Value Independence
          </h2>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto mb-16">
          <div className="text-center">
            <div className="text-4xl font-bold text-amber-600 mb-2">1,200+</div>
            <div className="text-slate-600">GitHub Stars</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-amber-600 mb-2">500+</div>
            <div className="text-slate-600">Active Installations</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-amber-600 mb-2">8,000+</div>
            <div className="text-slate-600">GPUs Orchestrated</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-green-600 mb-2">€0</div>
            <div className="text-slate-600">Avg Monthly Cost</div>
          </div>
        </div>

        {/* Testimonials */}
        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-full bg-gradient-to-br from-blue-400 to-blue-600"></div>
              <div>
                <div className="font-bold text-slate-900">Alex K.</div>
                <div className="text-sm text-slate-600">Solo Developer</div>
              </div>
            </div>
            <p className="text-slate-600 leading-relaxed">
              "I was spending $80/month on Claude for coding. Now I run Llama 70B on my gaming PC and old workstation.
              Same quality, $0 cost. Never going back."
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-full bg-gradient-to-br from-amber-400 to-amber-600"></div>
              <div>
                <div className="font-bold text-slate-900">Sarah M.</div>
                <div className="text-sm text-slate-600">CTO at StartupCo</div>
              </div>
            </div>
            <p className="text-slate-600 leading-relaxed">
              "We cut our AI costs from $500/month to zero by pooling our team's hardware. rbee just works.
              OpenAI-compatible API means no code changes."
            </p>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="h-12 w-12 rounded-full bg-gradient-to-br from-green-400 to-green-600"></div>
              <div>
                <div className="font-bold text-slate-900">Dr. Thomas R.</div>
                <div className="text-sm text-slate-600">Research Lab Director</div>
              </div>
            </div>
            <p className="text-slate-600 leading-relaxed">
              "GDPR compliance was killing us. rbee let us build AI infrastructure on-premises. EU-only routing with
              Rhai scripts. Perfect solution."
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
