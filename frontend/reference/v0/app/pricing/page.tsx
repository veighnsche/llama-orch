import { Button } from "@/components/ui/button"
import { Check, X } from "lucide-react"
import { EmailCapture } from "@/components/email-capture"

export default function PricingPage() {
  return (
    <div className="pt-16">
      {/* Hero Section */}
      <section className="py-24 bg-gradient-to-b from-slate-950 to-slate-900">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl lg:text-6xl font-bold text-white mb-6 text-balance">
              Start Free.
              <br />
              <span className="text-amber-500">Scale When Ready.</span>
            </h1>
            <p className="text-xl text-slate-300 leading-relaxed">
              All tiers include the full rbee orchestrator. No feature gates. No artificial limits. Just honest pricing
              for honest infrastructure.
            </p>
          </div>
        </div>
      </section>

      {/* Pricing Tiers */}
      <section className="py-24 bg-white">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {/* Tier 1: Free */}
            <div className="bg-white border-2 border-slate-200 rounded-lg p-8 space-y-6">
              <div>
                <h3 className="text-2xl font-bold text-slate-900">Home/Lab</h3>
                <div className="mt-4">
                  <span className="text-4xl font-bold text-slate-900">$0</span>
                  <span className="text-slate-600 ml-2">forever</span>
                </div>
              </div>

              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Unlimited GPUs</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">OpenAI-compatible API</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Multi-modal support</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Community support</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">GPL open source</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">CLI access</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Rhai scheduler</span>
                </li>
              </ul>

              <Button className="w-full bg-transparent" variant="outline">
                Download Now
              </Button>

              <p className="text-sm text-slate-600 text-center">For solo developers, hobbyists, homelab enthusiasts</p>
            </div>

            {/* Tier 2: Team (Most Popular) */}
            <div className="bg-amber-50 border-2 border-amber-500 rounded-lg p-8 space-y-6 relative">
              <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                <span className="bg-amber-500 text-white px-4 py-1 rounded-full text-sm font-medium">Most Popular</span>
              </div>

              <div>
                <h3 className="text-2xl font-bold text-slate-900">Team</h3>
                <div className="mt-4">
                  <span className="text-4xl font-bold text-slate-900">€99</span>
                  <span className="text-slate-600 ml-2">/month</span>
                </div>
                <p className="text-sm text-slate-600 mt-1">5-10 developers</p>
              </div>

              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-900 font-medium">Everything in Home/Lab</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Web UI management</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Team collaboration</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Priority support</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Rhai script templates</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Usage analytics</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Email support</span>
                </li>
              </ul>

              <Button className="w-full bg-amber-500 hover:bg-amber-600 text-slate-950">Start 30-Day Trial</Button>

              <p className="text-sm text-slate-600 text-center">For small teams, startups</p>
            </div>

            {/* Tier 3: Enterprise */}
            <div className="bg-white border-2 border-slate-200 rounded-lg p-8 space-y-6">
              <div>
                <h3 className="text-2xl font-bold text-slate-900">Enterprise</h3>
                <div className="mt-4">
                  <span className="text-4xl font-bold text-slate-900">Custom</span>
                </div>
                <p className="text-sm text-slate-600 mt-1">Contact sales</p>
              </div>

              <ul className="space-y-3">
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-900 font-medium">Everything in Team</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Dedicated instances</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Custom SLAs</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">White-label option</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Enterprise support</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">On-premises deployment</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-slate-600">Professional services</span>
                </li>
              </ul>

              <Button className="w-full bg-transparent" variant="outline">
                Contact Sales
              </Button>

              <p className="text-sm text-slate-600 text-center">For large teams, enterprises</p>
            </div>
          </div>
        </div>
      </section>

      {/* Feature Comparison Table */}
      <section className="py-24 bg-slate-50">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
              Detailed Feature Comparison
            </h2>
          </div>

          <div className="max-w-5xl mx-auto overflow-x-auto">
            <table className="w-full bg-white border border-slate-200 rounded-lg">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left p-4 font-semibold text-slate-900">Feature</th>
                  <th className="text-center p-4 font-semibold text-slate-900">Home/Lab</th>
                  <th className="text-center p-4 font-semibold text-slate-900 bg-amber-50">Team</th>
                  <th className="text-center p-4 font-semibold text-slate-900">Enterprise</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">Number of GPUs</td>
                  <td className="text-center p-4 text-slate-900">Unlimited</td>
                  <td className="text-center p-4 text-slate-900 bg-amber-50">Unlimited</td>
                  <td className="text-center p-4 text-slate-900">Unlimited</td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">OpenAI-compatible API</td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">Multi-GPU orchestration</td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">Rhai scheduler</td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">CLI access</td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">Web UI</td>
                  <td className="text-center p-4">
                    <X className="h-5 w-5 text-slate-300 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">Team collaboration</td>
                  <td className="text-center p-4">
                    <X className="h-5 w-5 text-slate-300 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">Support</td>
                  <td className="text-center p-4 text-slate-600">Community</td>
                  <td className="text-center p-4 text-slate-900 bg-amber-50">Priority Email</td>
                  <td className="text-center p-4 text-slate-900">Dedicated</td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">SLA</td>
                  <td className="text-center p-4">
                    <X className="h-5 w-5 text-slate-300 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <X className="h-5 w-5 text-slate-300 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
                <tr className="border-b border-slate-200">
                  <td className="p-4 text-slate-600">White-label</td>
                  <td className="text-center p-4">
                    <X className="h-5 w-5 text-slate-300 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <X className="h-5 w-5 text-slate-300 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
                <tr>
                  <td className="p-4 text-slate-600">Professional services</td>
                  <td className="text-center p-4">
                    <X className="h-5 w-5 text-slate-300 mx-auto" />
                  </td>
                  <td className="text-center p-4 bg-amber-50">
                    <X className="h-5 w-5 text-slate-300 mx-auto" />
                  </td>
                  <td className="text-center p-4">
                    <Check className="h-5 w-5 text-green-600 mx-auto" />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 bg-white">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">Pricing FAQs</h2>
          </div>

          <div className="max-w-3xl mx-auto space-y-6">
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-6">
              <h3 className="text-lg font-bold text-slate-900 mb-2">Is the free tier really free forever?</h3>
              <p className="text-slate-600 leading-relaxed">
                Yes. rbee is GPL open source. The Home/Lab tier is completely free with no time limits, no feature
                restrictions, and no hidden costs. You only pay for electricity to run your hardware.
              </p>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-6">
              <h3 className="text-lg font-bold text-slate-900 mb-2">What's the difference between tiers?</h3>
              <p className="text-slate-600 leading-relaxed">
                All tiers include the full rbee orchestrator with no feature gates. Paid tiers add convenience features
                like Web UI, team collaboration, priority support, and enterprise services. The core AI orchestration is
                identical across all tiers.
              </p>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-6">
              <h3 className="text-lg font-bold text-slate-900 mb-2">Can I upgrade or downgrade anytime?</h3>
              <p className="text-slate-600 leading-relaxed">
                Yes. You can upgrade to a paid tier anytime to access additional features. You can also downgrade back
                to the free tier without losing your data or configuration.
              </p>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-6">
              <h3 className="text-lg font-bold text-slate-900 mb-2">Do you offer discounts for non-profits?</h3>
              <p className="text-slate-600 leading-relaxed">
                Yes. We offer 50% discounts for registered non-profits, educational institutions, and open source
                projects. Contact sales for details.
              </p>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-6">
              <h3 className="text-lg font-bold text-slate-900 mb-2">What payment methods do you accept?</h3>
              <p className="text-slate-600 leading-relaxed">
                We accept credit cards, bank transfers, and purchase orders for enterprise customers. All payments are
                processed securely through Stripe.
              </p>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-lg p-6">
              <h3 className="text-lg font-bold text-slate-900 mb-2">Is there a trial period?</h3>
              <p className="text-slate-600 leading-relaxed">
                The Team tier includes a 30-day free trial with full access to all features. No credit card required to
                start. Enterprise customers can request a custom trial period.
              </p>
            </div>
          </div>
        </div>
      </section>

      <EmailCapture />
    </div>
  )
}
