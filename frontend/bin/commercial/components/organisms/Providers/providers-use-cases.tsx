import { Gamepad2, Server, Cpu, Monitor } from 'lucide-react'

export function ProvidersUseCases() {
  return (
    <section className="border-b border-border bg-gradient-to-b from-background to-card px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">Who's Earning with rbee?</h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
            From gamers to homelab enthusiasts, anyone with a GPU can turn idle hardware into income.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
                <Gamepad2 className="h-7 w-7 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Gaming PC Owners</h3>
                <div className="text-sm text-muted-foreground">Most common provider type</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              "I game for 3-4 hours a day. The rest of the time, my RTX 4090 just sits there. Now it earns me €150/month
              while I'm at work or sleeping."
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-muted-foreground">
                <span>Typical GPU:</span>
                <span className="text-foreground">RTX 4080/4090</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>Availability:</span>
                <span className="text-foreground">16-20 hours/day</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>Monthly earnings:</span>
                <span className="font-bold text-primary">€120-180</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
                <Server className="h-7 w-7 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Homelab Enthusiasts</h3>
                <div className="text-sm text-muted-foreground">Multiple GPUs, high earnings</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              "I have 4 GPUs across my homelab. They were mostly idle. Now they earn €400/month combined. Pays for my
              entire homelab electricity bill plus profit."
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-muted-foreground">
                <span>Typical setup:</span>
                <span className="text-foreground">3-6 GPUs</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>Availability:</span>
                <span className="text-foreground">20-24 hours/day</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>Monthly earnings:</span>
                <span className="font-bold text-primary">€300-600</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
                <Cpu className="h-7 w-7 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Former Crypto Miners</h3>
                <div className="text-sm text-muted-foreground">Repurpose mining rigs</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              "After Ethereum went proof-of-stake, my mining rig was useless. Now it earns more with rbee than it ever
              did mining. Better margins, too."
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-muted-foreground">
                <span>Typical setup:</span>
                <span className="text-foreground">6-12 GPUs</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>Availability:</span>
                <span className="text-foreground">24 hours/day</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>Monthly earnings:</span>
                <span className="font-bold text-primary">€600-1,200</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
                <Monitor className="h-7 w-7 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-foreground">Workstation Owners</h3>
                <div className="text-sm text-muted-foreground">Professional GPUs earning</div>
              </div>
            </div>
            <p className="mb-4 text-pretty leading-relaxed text-muted-foreground">
              "I'm a 3D artist. My workstation has an RTX 4080 that's only busy during renders. The rest of the time, it
              earns €100/month on rbee."
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-muted-foreground">
                <span>Typical GPU:</span>
                <span className="text-foreground">RTX 4070-4080</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>Availability:</span>
                <span className="text-foreground">12-16 hours/day</span>
              </div>
              <div className="flex justify-between text-muted-foreground">
                <span>Monthly earnings:</span>
                <span className="font-bold text-primary">€80-140</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
