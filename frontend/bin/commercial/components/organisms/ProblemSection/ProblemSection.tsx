import { AlertTriangle, DollarSign, Lock, ArrowRight } from 'lucide-react'
import { SectionContainer, FeatureCard } from '@/components/molecules'
import Image from 'next/image'

export function ProblemSection() {
  return (
    <SectionContainer 
      title="The Hidden Cost of AI Dependency"
      subtitle="When vendors change the rules, your roadmap pays the price."
    >
      {/* Vendor lock-in illustration */}
      <Image
        src="/illustrations/vendor-lock-in.svg"
        width={1200}
        height={180}
        priority
        className="mx-auto mb-8 opacity-90 max-w-full h-auto"
        alt="Two infrastructure islands separated by a paywall gate with broken cables and caution beacons, illustrating vendor lock-in and rising costs"
      />

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-7 md:gap-8 max-w-6xl mx-auto px-6">
        <FeatureCard
          icon={AlertTriangle}
          title="Loss of Control"
          description="Vendors deprecate, rate-limit, or shut down. Your workflows and uptime hinge on choices you don't control."
          iconColor="chart-4"
          size="lg"
          className="h-full border-border/70 hover:border-primary/40 hover:bg-card/80 transition-colors animate-in fade-in slide-in-from-bottom-2 duration-500 [animation-delay:100ms]"
        />
        <FeatureCard
          icon={DollarSign}
          title="Unpredictable Costs"
          description="Pricing shifts without notice. As usage scales, bills spike—and what started cheap becomes unsustainable."
          iconColor="chart-2"
          size="lg"
          className="h-full border-border/70 hover:border-primary/40 hover:bg-card/80 transition-colors animate-in fade-in slide-in-from-bottom-2 duration-500 [animation-delay:200ms]"
        />
        <FeatureCard
          icon={Lock}
          title="Privacy & Compliance Risks"
          description="Sensitive data exits your network. Cloud dependencies complicate audits and widen regulatory exposure."
          iconColor="chart-3"
          size="lg"
          className="h-full border-border/70 hover:border-primary/40 hover:bg-card/80 transition-colors animate-in fade-in slide-in-from-bottom-2 duration-500 [animation-delay:300ms]"
        />
      </div>

      <div className="max-w-3xl mx-auto text-center mt-10">
        <p className="text-lg text-muted-foreground leading-relaxed text-balance">
          Whether you're building with AI, monetizing hardware, or meeting compliance—outsourcing core models creates risk you can't budget for.
        </p>
        <a 
          href="#solution" 
          className="mt-4 inline-flex items-center gap-2 text-primary hover:text-primary/90 font-medium transition-colors"
        >
          See how rbee restores control
          <ArrowRight aria-hidden="true" className="h-4 w-4" />
        </a>
      </div>
    </SectionContainer>
  )
}
