import { Button } from "@/components/ui/button"
import { Calendar, FileText, MessageSquare } from "lucide-react"

export function EnterpriseCTA() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-slate-950 via-amber-950/10 to-slate-950 px-6 py-24">
      <div className="mx-auto max-w-4xl text-center">
        <h2 className="mb-4 text-4xl font-bold text-white lg:text-5xl">Ready to Meet Your Compliance Requirements?</h2>
        <p className="mb-12 text-balance text-xl text-slate-300">
          Schedule a demo with our compliance team or download our compliance documentation package.
        </p>

        <div className="mb-12 grid gap-6 md:grid-cols-3">
          {/* Option 1 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-6">
            <div className="mb-4 flex justify-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <Calendar className="h-6 w-6 text-amber-400" />
              </div>
            </div>
            <h3 className="mb-2 font-semibold text-white">Schedule Demo</h3>
            <p className="mb-4 text-sm leading-relaxed text-slate-400">
              30-minute demo with our compliance team. See rbee in action.
            </p>
            <Button className="w-full bg-amber-500 text-slate-950 hover:bg-amber-400">Book Demo</Button>
          </div>

          {/* Option 2 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-6">
            <div className="mb-4 flex justify-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <FileText className="h-6 w-6 text-amber-400" />
              </div>
            </div>
            <h3 className="mb-2 font-semibold text-white">Compliance Pack</h3>
            <p className="mb-4 text-sm leading-relaxed text-slate-400">
              Download GDPR, SOC2, and ISO 27001 documentation.
            </p>
            <Button variant="outline" className="w-full border-slate-700 text-white hover:bg-slate-800 bg-transparent">
              Download Docs
            </Button>
          </div>

          {/* Option 3 */}
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-6">
            <div className="mb-4 flex justify-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-amber-500/10">
                <MessageSquare className="h-6 w-6 text-amber-400" />
              </div>
            </div>
            <h3 className="mb-2 font-semibold text-white">Talk to Sales</h3>
            <p className="mb-4 text-sm leading-relaxed text-slate-400">
              Discuss your specific compliance requirements.
            </p>
            <Button variant="outline" className="w-full border-slate-700 text-white hover:bg-slate-800 bg-transparent">
              Contact Sales
            </Button>
          </div>
        </div>

        <p className="text-sm text-slate-400">
          Enterprise support available 24/7. Typical deployment: 6-8 weeks from consultation to production.
        </p>
      </div>
    </section>
  )
}
