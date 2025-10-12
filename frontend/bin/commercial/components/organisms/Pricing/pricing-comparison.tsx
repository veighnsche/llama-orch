import { Check, X } from 'lucide-react'
import { SectionContainer } from '@/components/molecules'

export function PricingComparison() {
  return (
    <SectionContainer title="Detailed Feature Comparison" bgVariant="secondary">
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
    </SectionContainer>
  )
}
