import { Check, X } from "lucide-react"

export function PricingComparison() {
  return (
    <section className="py-24 bg-secondary">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
            Detailed Feature Comparison
          </h2>
        </div>

        <div className="max-w-5xl mx-auto overflow-x-auto">
          <table className="w-full bg-card border border-border rounded-lg">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left p-4 font-semibold text-foreground">Feature</th>
                <th className="text-center p-4 font-semibold text-foreground">Home/Lab</th>
                <th className="text-center p-4 font-semibold text-foreground bg-primary/5">Team</th>
                <th className="text-center p-4 font-semibold text-foreground">Enterprise</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">Number of GPUs</td>
                <td className="text-center p-4 text-foreground">Unlimited</td>
                <td className="text-center p-4 text-foreground bg-primary/5">Unlimited</td>
                <td className="text-center p-4 text-foreground">Unlimited</td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">OpenAI-compatible API</td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">Multi-GPU orchestration</td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">Rhai scheduler</td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">CLI access</td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">Web UI</td>
                <td className="text-center p-4">
                  <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">Team collaboration</td>
                <td className="text-center p-4">
                  <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">Support</td>
                <td className="text-center p-4 text-muted-foreground">Community</td>
                <td className="text-center p-4 text-foreground bg-primary/5">Priority Email</td>
                <td className="text-center p-4 text-foreground">Dedicated</td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">SLA</td>
                <td className="text-center p-4">
                  <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 text-muted-foreground">White-label</td>
                <td className="text-center p-4">
                  <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr>
                <td className="p-4 text-muted-foreground">Professional services</td>
                <td className="text-center p-4">
                  <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                </td>
                <td className="text-center p-4 bg-primary/5">
                  <X className="h-5 w-5 text-muted-foreground/30 mx-auto" />
                </td>
                <td className="text-center p-4">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
