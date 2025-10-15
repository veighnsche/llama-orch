export function WhatIsRbee() {
  return (
    <section className="py-16 bg-slate-50">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="text-center space-y-6">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900">What is rbee?</h2>

          <p className="text-xl text-slate-700 leading-relaxed">
            <span className="font-semibold text-slate-900">rbee</span> (pronounced "are-bee") is an{" "}
            <span className="font-semibold text-amber-600">open-source AI orchestration platform</span> that turns all
            the computers in your home or office network into a unified AI infrastructure.
          </p>

          <div className="grid md:grid-cols-3 gap-6 pt-8">
            <div className="bg-white p-6 rounded-lg border border-slate-200">
              <div className="text-4xl font-bold text-amber-600 mb-2">$0</div>
              <div className="text-sm text-slate-600">Monthly costs after setup. Just electricity.</div>
            </div>

            <div className="bg-white p-6 rounded-lg border border-slate-200">
              <div className="text-4xl font-bold text-amber-600 mb-2">100%</div>
              <div className="text-sm text-slate-600">Private. Your code and data never leave your network.</div>
            </div>

            <div className="bg-white p-6 rounded-lg border border-slate-200">
              <div className="text-4xl font-bold text-amber-600 mb-2">All</div>
              <div className="text-sm text-slate-600">Your GPUs working together—CUDA, Metal, CPU.</div>
            </div>
          </div>

          <p className="text-lg text-slate-600 pt-4">
            Whether you're a developer building AI tools, someone with idle GPUs to monetize, or an enterprise needing
            compliant AI infrastructure—rbee gives you complete control.
          </p>
        </div>
      </div>
    </section>
  )
}
