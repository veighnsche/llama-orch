import { Gamepad2, Server, Cpu, Monitor } from "lucide-react"

export function ProvidersUseCases() {
  return (
    <section className="border-b border-slate-800 bg-gradient-to-b from-slate-950 to-slate-900 px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-white lg:text-5xl">Who's Earning with rbee?</h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-slate-300">
            From gamers to homelab enthusiasts, anyone with a GPU can turn idle hardware into income.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-amber-500/10">
                <Gamepad2 className="h-7 w-7 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Gaming PC Owners</h3>
                <div className="text-sm text-slate-400">Most common provider type</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              "I game for 3-4 hours a day. The rest of the time, my RTX 4090 just sits there. Now it earns me €150/month
              while I'm at work or sleeping."
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-slate-400">
                <span>Typical GPU:</span>
                <span className="text-white">RTX 4080/4090</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Availability:</span>
                <span className="text-white">16-20 hours/day</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Monthly earnings:</span>
                <span className="font-bold text-amber-400">€120-180</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-amber-500/10">
                <Server className="h-7 w-7 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Homelab Enthusiasts</h3>
                <div className="text-sm text-slate-400">Multiple GPUs, high earnings</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              "I have 4 GPUs across my homelab. They were mostly idle. Now they earn €400/month combined. Pays for my
              entire homelab electricity bill plus profit."
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-slate-400">
                <span>Typical setup:</span>
                <span className="text-white">3-6 GPUs</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Availability:</span>
                <span className="text-white">20-24 hours/day</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Monthly earnings:</span>
                <span className="font-bold text-amber-400">€300-600</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-amber-500/10">
                <Cpu className="h-7 w-7 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Former Crypto Miners</h3>
                <div className="text-sm text-slate-400">Repurpose mining rigs</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              "After Ethereum went proof-of-stake, my mining rig was useless. Now it earns more with rbee than it ever
              did mining. Better margins, too."
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-slate-400">
                <span>Typical setup:</span>
                <span className="text-white">6-12 GPUs</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Availability:</span>
                <span className="text-white">24 hours/day</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Monthly earnings:</span>
                <span className="font-bold text-amber-400">€600-1,200</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-amber-500/10">
                <Monitor className="h-7 w-7 text-amber-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">Workstation Owners</h3>
                <div className="text-sm text-slate-400">Professional GPUs earning</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-slate-300">
              "I'm a 3D artist. My workstation has an RTX 4080 that's only busy during renders. The rest of the time, it
              earns €100/month on rbee."
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-slate-400">
                <span>Typical GPU:</span>
                <span className="text-white">RTX 4070-4080</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Availability:</span>
                <span className="text-white">12-16 hours/day</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Monthly earnings:</span>
                <span className="font-bold text-amber-400">€80-140</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
