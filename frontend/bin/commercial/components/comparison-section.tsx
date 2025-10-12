import { Check, X } from "lucide-react"

export function ComparisonSection() {
  return (
    <section className="py-24 bg-secondary">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
            Why Developers Choose rbee
          </h2>
        </div>

        <div className="max-w-6xl mx-auto overflow-x-auto">
          <table className="w-full bg-card border border-border rounded-lg">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left p-4 font-bold text-card-foreground">Feature</th>
                <th className="p-4 font-bold text-primary bg-primary/5">rbee</th>
                <th className="p-4 font-medium text-muted-foreground">OpenAI/Anthropic</th>
                <th className="p-4 font-medium text-muted-foreground">Ollama</th>
                <th className="p-4 font-medium text-muted-foreground">Runpod/Vast.ai</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border">
                <td className="p-4 font-medium text-card-foreground">Cost</td>
                <td className="p-4 text-center bg-primary/5">
                  <div className="text-sm font-medium text-card-foreground">$0</div>
                  <div className="text-xs text-muted-foreground">(your hardware)</div>
                </td>
                <td className="p-4 text-center text-sm text-muted-foreground">$20-100/mo per dev</td>
                <td className="p-4 text-center text-sm text-muted-foreground">$0</td>
                <td className="p-4 text-center text-sm text-muted-foreground">$0.50-2/hr</td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 font-medium text-card-foreground">Privacy</td>
                <td className="p-4 text-center bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                  <div className="text-xs text-muted-foreground mt-1">Complete</div>
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-destructive mx-auto" />
                  <div className="text-xs text-muted-foreground mt-1">Limited</div>
                </td>
                <td className="p-4 text-center">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                  <div className="text-xs text-muted-foreground mt-1">Complete</div>
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-destructive mx-auto" />
                  <div className="text-xs text-muted-foreground mt-1">Limited</div>
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 font-medium text-card-foreground">Multi-GPU</td>
                <td className="p-4 text-center bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                  <div className="text-xs text-muted-foreground mt-1">Orchestrated</div>
                </td>
                <td className="p-4 text-center text-sm text-muted-foreground">N/A</td>
                <td className="p-4 text-center">
                  <div className="text-sm text-muted-foreground">Limited</div>
                </td>
                <td className="p-4 text-center">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 font-medium text-card-foreground">OpenAI API</td>
                <td className="p-4 text-center bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="p-4 text-center">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                </td>
                <td className="p-4 text-center">
                  <div className="text-sm text-muted-foreground">Partial</div>
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-destructive mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-border">
                <td className="p-4 font-medium text-card-foreground">Custom Routing</td>
                <td className="p-4 text-center bg-primary/5">
                  <Check className="h-5 w-5 text-chart-3 mx-auto" />
                  <div className="text-xs text-muted-foreground mt-1">Rhai scripts</div>
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-destructive mx-auto" />
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-destructive mx-auto" />
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-destructive mx-auto" />
                </td>
              </tr>
              <tr>
                <td className="p-4 font-medium text-card-foreground">Rate Limits</td>
                <td className="p-4 text-center bg-primary/5">
                  <div className="text-sm font-medium text-chart-3">None</div>
                </td>
                <td className="p-4 text-center">
                  <div className="text-sm text-destructive">Yes</div>
                </td>
                <td className="p-4 text-center">
                  <div className="text-sm font-medium text-chart-3">None</div>
                </td>
                <td className="p-4 text-center">
                  <div className="text-sm text-destructive">Yes</div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
