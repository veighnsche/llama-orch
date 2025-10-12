import { SectionContainer, StatCard } from '@/components/molecules'

export function WhatIsRbee() {
  return (
    <SectionContainer 
      title="What is rbee?"
      bgVariant="secondary"
      maxWidth="4xl"
    >

      <p className="text-xl text-foreground leading-relaxed text-center">
        <span className="font-semibold text-card-foreground">rbee</span> (pronounced "are-bee") is an{" "}
        <span className="font-semibold text-primary">open source AI orchestration platform</span> that turns all
        the computers in your home or office network into a unified AI infrastructure.
      </p>

      <div className="grid md:grid-cols-3 gap-6 pt-8">
        <div className="bg-card p-6 rounded-lg border border-border">
          <StatCard value="$0" label="Monthly costs after setup. Just electricity." size="lg" />
        </div>
        <div className="bg-card p-6 rounded-lg border border-border">
          <StatCard value="100%" label="Private. Your code and data never leave your network." size="lg" />
        </div>
        <div className="bg-card p-6 rounded-lg border border-border">
          <StatCard value="All" label="Your GPUs working together—CUDA, Metal, CPU." size="lg" />
        </div>
      </div>

      <p className="text-lg text-muted-foreground pt-4 text-center">
        Whether you're a developer building AI tools, someone with idle GPUs to monetize, or an enterprise needing
        compliant AI infrastructure—rbee gives you complete control.
      </p>
    </SectionContainer>
  )
}
