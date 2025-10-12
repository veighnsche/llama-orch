export function WhatIsRbee() {
  return (
    <section className="py-16 bg-secondary">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="text-center space-y-6">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground">What is rbee?</h2>

          <p className="text-xl text-foreground leading-relaxed">
            <span className="font-semibold text-card-foreground">rbee</span> (pronounced "are-bee") is an{" "}
            <span className="font-semibold text-primary">open-source AI orchestration platform</span> that turns all
            the computers in your home or office network into a unified AI infrastructure.
          </p>

          <div className="grid md:grid-cols-3 gap-6 pt-8">
            <div className="bg-card p-6 rounded-lg border border-border">
              <div className="text-4xl font-bold text-primary mb-2">$0</div>
              <div className="text-sm text-muted-foreground">Monthly costs after setup. Just electricity.</div>
            </div>

            <div className="bg-card p-6 rounded-lg border border-border">
              <div className="text-4xl font-bold text-primary mb-2">100%</div>
              <div className="text-sm text-muted-foreground">Private. Your code and data never leave your network.</div>
            </div>

            <div className="bg-card p-6 rounded-lg border border-border">
              <div className="text-4xl font-bold text-primary mb-2">All</div>
              <div className="text-sm text-muted-foreground">Your GPUs working together—CUDA, Metal, CPU.</div>
            </div>
          </div>

          <p className="text-lg text-muted-foreground pt-4">
            Whether you're a developer building AI tools, someone with idle GPUs to monetize, or an enterprise needing
            compliant AI infrastructure—rbee gives you complete control.
          </p>
        </div>
      </div>
    </section>
  )
}
