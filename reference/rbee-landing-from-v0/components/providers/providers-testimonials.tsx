import { Star } from "lucide-react"

export function ProvidersTestimonials() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-slate-950 to-slate-900 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-white lg:text-5xl">What Providers Are Saying</h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-slate-300">
            Real earnings from real GPU providers on the rbee marketplace.
          </p>
        </div>

        <div className="mb-12 grid gap-8 md:grid-cols-3">
          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-4 flex gap-1">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="h-5 w-5 fill-amber-400 text-amber-400" />
              ))}
            </div>
            <p className="mb-6 text-pretty leading-relaxed text-slate-300">
              "My RTX 4090 was sitting idle 20 hours a day. Now it earns me €160/month. Literally passive income while I
              sleep. Setup took 10 minutes."
            </p>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-amber-500 to-orange-500" />
              <div>
                <div className="font-medium text-white">Marcus T.</div>
                <div className="text-sm text-slate-400">Gaming PC Owner • €160/mo</div>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-4 flex gap-1">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="h-5 w-5 fill-amber-400 text-amber-400" />
              ))}
            </div>
            <p className="mb-6 text-pretty leading-relaxed text-slate-300">
              "I have 4 GPUs in my homelab. They were mostly idle. Now they earn €420/month combined. Pays for my entire
              homelab electricity plus profit."
            </p>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-amber-500 to-orange-500" />
              <div>
                <div className="font-medium text-white">Sarah K.</div>
                <div className="text-sm text-slate-400">Homelab Enthusiast • €420/mo</div>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-4 flex gap-1">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="h-5 w-5 fill-amber-400 text-amber-400" />
              ))}
            </div>
            <p className="mb-6 text-pretty leading-relaxed text-slate-300">
              "After Ethereum went proof-of-stake, my mining rig was useless. Now it earns more with rbee than it ever
              did mining. Better margins, too."
            </p>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-amber-500 to-orange-500" />
              <div>
                <div className="font-medium text-white">David L.</div>
                <div className="text-sm text-slate-400">Former Miner • €780/mo</div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid gap-8 md:grid-cols-4">
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-amber-400">500+</div>
            <div className="text-sm text-slate-400">Active Providers</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-amber-400">2,000+</div>
            <div className="text-sm text-slate-400">GPUs Earning</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-amber-400">€180K+</div>
            <div className="text-sm text-slate-400">Paid to Providers</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-amber-400">4.8/5</div>
            <div className="text-sm text-slate-400">Average Rating</div>
          </div>
        </div>
      </div>
    </section>
  )
}
