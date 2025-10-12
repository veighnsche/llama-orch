import { Star } from 'lucide-react'

export function ProvidersTestimonials() {
  return (
    <section className="border-b border-border bg-gradient-to-b from-background to-card px-6 py-24">
      <div className="mx-auto max-w-7xl">
        <div className="mb-16 text-center">
          <h2 className="mb-4 text-balance text-4xl font-bold text-foreground lg:text-5xl">
            What Providers Are Saying
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-xl text-muted-foreground">
            Real earnings from real GPU providers on the rbee marketplace.
          </p>
        </div>

        <div className="mb-12 grid gap-8 md:grid-cols-3">
          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-4 flex gap-1">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="h-5 w-5 fill-amber-400 text-primary" />
              ))}
            </div>
            <p className="mb-6 text-pretty leading-relaxed text-muted-foreground">
              "My RTX 4090 was sitting idle 20 hours a day. Now it earns me €160/month. Literally passive income while I
              sleep. Setup took 10 minutes."
            </p>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-amber-500 to-orange-500" />
              <div>
                <div className="font-medium text-foreground">Marcus T.</div>
                <div className="text-sm text-muted-foreground">Gaming PC Owner • €160/mo</div>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-4 flex gap-1">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="h-5 w-5 fill-amber-400 text-primary" />
              ))}
            </div>
            <p className="mb-6 text-pretty leading-relaxed text-muted-foreground">
              "I have 4 GPUs in my homelab. They were mostly idle. Now they earn €420/month combined. Pays for my entire
              homelab electricity plus profit."
            </p>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-amber-500 to-orange-500" />
              <div>
                <div className="font-medium text-foreground">Sarah K.</div>
                <div className="text-sm text-muted-foreground">Homelab Enthusiast • €420/mo</div>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-border bg-gradient-to-b from-card to-background p-8">
            <div className="mb-4 flex gap-1">
              {[...Array(5)].map((_, i) => (
                <Star key={i} className="h-5 w-5 fill-amber-400 text-primary" />
              ))}
            </div>
            <p className="mb-6 text-pretty leading-relaxed text-muted-foreground">
              "After Ethereum went proof-of-stake, my mining rig was useless. Now it earns more with rbee than it ever
              did mining. Better margins, too."
            </p>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-gradient-to-br from-amber-500 to-orange-500" />
              <div>
                <div className="font-medium text-foreground">David L.</div>
                <div className="text-sm text-muted-foreground">Former Miner • €780/mo</div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid gap-8 md:grid-cols-4">
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-primary">500+</div>
            <div className="text-sm text-muted-foreground">Active Providers</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-primary">2,000+</div>
            <div className="text-sm text-muted-foreground">GPUs Earning</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-primary">€180K+</div>
            <div className="text-sm text-muted-foreground">Paid to Providers</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-4xl font-bold text-primary">4.8/5</div>
            <div className="text-sm text-muted-foreground">Average Rating</div>
          </div>
        </div>
      </div>
    </section>
  )
}
