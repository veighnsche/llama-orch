import { Check, X } from "lucide-react"

export function EnterpriseComparison() {
  return (
    <section className="border-b border-slate-800 bg-slate-950 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-4xl font-bold text-white">Why Enterprises Choose rbee</h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-slate-300">
            Compare rbee's compliance and security features against external AI providers.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-slate-800">
                <th className="p-4 text-left text-sm font-semibold text-slate-400">Feature</th>
                <th className="bg-amber-500/5 p-4 text-center text-sm font-semibold text-amber-400">
                  rbee (Self-Hosted)
                </th>
                <th className="p-4 text-center text-sm font-semibold text-slate-400">OpenAI / Anthropic</th>
                <th className="p-4 text-center text-sm font-semibold text-slate-400">Azure OpenAI</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  feature: "Data Sovereignty",
                  rbee: true,
                  openai: false,
                  azure: "Partial",
                },
                {
                  feature: "EU-Only Deployment",
                  rbee: true,
                  openai: false,
                  azure: "Partial",
                },
                {
                  feature: "GDPR Compliant",
                  rbee: true,
                  openai: "Partial",
                  azure: "Partial",
                },
                {
                  feature: "Immutable Audit Logs",
                  rbee: true,
                  openai: false,
                  azure: false,
                },
                {
                  feature: "7-Year Audit Retention",
                  rbee: true,
                  openai: false,
                  azure: false,
                },
                {
                  feature: "SOC2 Type II Ready",
                  rbee: true,
                  openai: true,
                  azure: true,
                },
                {
                  feature: "ISO 27001 Aligned",
                  rbee: true,
                  openai: true,
                  azure: true,
                },
                {
                  feature: "Zero US Cloud Dependencies",
                  rbee: true,
                  openai: false,
                  azure: false,
                },
                {
                  feature: "On-Premises Deployment",
                  rbee: true,
                  openai: false,
                  azure: false,
                },
                {
                  feature: "Complete Control",
                  rbee: true,
                  openai: false,
                  azure: "Partial",
                },
                {
                  feature: "Custom SLAs",
                  rbee: true,
                  openai: false,
                  azure: true,
                },
                {
                  feature: "White-Label Option",
                  rbee: true,
                  openai: false,
                  azure: false,
                },
              ].map((row, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="p-4 text-sm text-slate-300">{row.feature}</td>
                  <td className="bg-amber-500/5 p-4 text-center">
                    {row.rbee === true ? (
                      <Check className="mx-auto h-5 w-5 text-green-400" />
                    ) : row.rbee === false ? (
                      <X className="mx-auto h-5 w-5 text-red-400" />
                    ) : (
                      <span className="text-sm text-amber-400">{row.rbee}</span>
                    )}
                  </td>
                  <td className="p-4 text-center">
                    {row.openai === true ? (
                      <Check className="mx-auto h-5 w-5 text-green-400" />
                    ) : row.openai === false ? (
                      <X className="mx-auto h-5 w-5 text-red-400" />
                    ) : (
                      <span className="text-sm text-slate-400">{row.openai}</span>
                    )}
                  </td>
                  <td className="p-4 text-center">
                    {row.azure === true ? (
                      <Check className="mx-auto h-5 w-5 text-green-400" />
                    ) : row.azure === false ? (
                      <X className="mx-auto h-5 w-5 text-red-400" />
                    ) : (
                      <span className="text-sm text-slate-400">{row.azure}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-8 text-center text-sm text-slate-400">
          * Comparison based on publicly available information as of October 2025
        </div>
      </div>
    </section>
  )
}
