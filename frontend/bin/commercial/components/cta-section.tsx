import { Button } from "@/components/ui/button"
import { ArrowRight, BookOpen, MessageCircle } from "lucide-react"

export function CTASection() {
  return (
    <section className="py-24 bg-gradient-to-br from-slate-950 via-slate-900 to-amber-950">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          <h2 className="text-4xl lg:text-6xl font-bold text-white mb-6 text-balance">
            Take Control of Your
            <br />
            <span className="text-amber-400">AI Infrastructure Today.</span>
          </h2>

          <p className="text-xl text-slate-300 leading-relaxed">
            Join hundreds of users, providers, and enterprises who've chosen independence.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              size="lg"
              className="bg-amber-500 hover:bg-amber-600 text-slate-950 font-semibold text-lg h-14 px-8"
            >
              Get Started Free
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-slate-600 text-slate-200 hover:bg-slate-800 h-14 px-8 bg-transparent"
            >
              <BookOpen className="mr-2 h-5 w-5" />
              View Documentation
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-slate-600 text-slate-200 hover:bg-slate-800 h-14 px-8 bg-transparent"
            >
              <MessageCircle className="mr-2 h-5 w-5" />
              Join Discord
            </Button>
          </div>

          <p className="text-slate-400 pt-4">100% open source. No credit card required. Install in 15 minutes.</p>
        </div>
      </div>
    </section>
  )
}
