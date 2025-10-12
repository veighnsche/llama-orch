import { Check, X } from "lucide-react"

export function ComparisonSection() {
  return (
    <section className="py-24 bg-slate-50">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
            Why Developers Choose rbee
          </h2>
        </div>

        <div className="max-w-6xl mx-auto overflow-x-auto">
          <table className="w-full bg-white border border-slate-200 rounded-lg">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left p-4 font-bold text-slate-900">Feature</th>
                <th className="p-4 font-bold text-amber-600 bg-amber-50">rbee</th>
                <th className="p-4 font-medium text-slate-600">OpenAI/Anthropic</th>
                <th className="p-4 font-medium text-slate-600">Ollama</th>
                <th className="p-4 font-medium text-slate-600">Runpod/Vast.ai</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-slate-200">
                <td className="p-4 font-medium text-slate-900">Cost</td>
                <td className="p-4 text-center bg-amber-50">
                  <div className="text-sm font-medium text-slate-900">$0</div>
                  <div className="text-xs text-slate-600">(your hardware)</div>
                </td>
                <td className="p-4 text-center text-sm text-slate-600">$20-100/mo per dev</td>
                <td className="p-4 text-center text-sm text-slate-600">$0</td>
                <td className="p-4 text-center text-sm text-slate-600">$0.50-2/hr</td>
              </tr>
              <tr className="border-b border-slate-200">
                <td className="p-4 font-medium text-slate-900">Privacy</td>
                <td className="p-4 text-center bg-amber-50">
                  <Check className="h-5 w-5 text-green-600 mx-auto" />
                  <div className="text-xs text-slate-600 mt-1">Complete</div>
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-red-500 mx-auto" />
                  <div className="text-xs text-slate-600 mt-1">Limited</div>
                </td>
                <td className="p-4 text-center">
                  <Check className="h-5 w-5 text-green-600 mx-auto" />
                  <div className="text-xs text-slate-600 mt-1">Complete</div>
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-red-500 mx-auto" />
                  <div className="text-xs text-slate-600 mt-1">Limited</div>
                </td>
              </tr>
              <tr className="border-b border-slate-200">
                <td className="p-4 font-medium text-slate-900">Multi-GPU</td>
                <td className="p-4 text-center bg-amber-50">
                  <Check className="h-5 w-5 text-green-600 mx-auto" />
                  <div className="text-xs text-slate-600 mt-1">Orchestrated</div>
                </td>
                <td className="p-4 text-center text-sm text-slate-600">N/A</td>
                <td className="p-4 text-center">
                  <div className="text-sm text-slate-600">Limited</div>
                </td>
                <td className="p-4 text-center">
                  <Check className="h-5 w-5 text-green-600 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-slate-200">
                <td className="p-4 font-medium text-slate-900">OpenAI API</td>
                <td className="p-4 text-center bg-amber-50">
                  <Check className="h-5 w-5 text-green-600 mx-auto" />
                </td>
                <td className="p-4 text-center">
                  <Check className="h-5 w-5 text-green-600 mx-auto" />
                </td>
                <td className="p-4 text-center">
                  <div className="text-sm text-slate-600">Partial</div>
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-red-500 mx-auto" />
                </td>
              </tr>
              <tr className="border-b border-slate-200">
                <td className="p-4 font-medium text-slate-900">Custom Routing</td>
                <td className="p-4 text-center bg-amber-50">
                  <Check className="h-5 w-5 text-green-600 mx-auto" />
                  <div className="text-xs text-slate-600 mt-1">Rhai scripts</div>
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-red-500 mx-auto" />
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-red-500 mx-auto" />
                </td>
                <td className="p-4 text-center">
                  <X className="h-5 w-5 text-red-500 mx-auto" />
                </td>
              </tr>
              <tr>
                <td className="p-4 font-medium text-slate-900">Rate Limits</td>
                <td className="p-4 text-center bg-amber-50">
                  <div className="text-sm font-medium text-green-600">None</div>
                </td>
                <td className="p-4 text-center">
                  <div className="text-sm text-red-600">Yes</div>
                </td>
                <td className="p-4 text-center">
                  <div className="text-sm font-medium text-green-600">None</div>
                </td>
                <td className="p-4 text-center">
                  <div className="text-sm text-red-600">Yes</div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
