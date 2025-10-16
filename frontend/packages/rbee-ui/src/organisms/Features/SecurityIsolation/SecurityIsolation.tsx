import { Card, CardContent } from '@rbee/ui/atoms'
import { BulletListItem, CrateCard, IconCardHeader, SectionContainer } from '@rbee/ui/molecules'
import { Lock, Shield } from 'lucide-react'
import { SECURITY_CRATES } from './securityCratesData'

export function SecurityIsolation() {
  return (
    <SectionContainer
      title="Security & Isolation"
      bgVariant="background"
      subtitle="Defense-in-depth with six focused Rust crates. Enterprise-grade security for your homelab."
    >
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Block 1: Crate Lattice */}
        <Card className="animate-in fade-in slide-in-from-bottom-2">
          <CardContent className="p-6 md:p-8">
            <IconCardHeader
              icon={Shield}
              iconTone="chart-2"
              iconSize="md"
              title="Six Specialized Security Crates"
              subtitle="Each concern ships as its own Rust crate—focused responsibility, no monolith."
              useCardHeader={false}
              className="mb-6"
            />

            {/* Crate lattice grid */}
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {SECURITY_CRATES.map((crate) => (
                <CrateCard
                  key={crate.name}
                  name={crate.name}
                  description={crate.description}
                  hoverColor={crate.hoverColor}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Block 2: Process Isolation */}
        <div className="grid md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-2 delay-100">
          <Card>
            <CardContent className="p-6">
              <IconCardHeader
                icon={Lock}
                iconTone="chart-3"
                iconSize="sm"
                title="Process Isolation"
                subtitle="Workers run in isolated processes with clean shutdown."
                titleClassName="text-lg"
                subtitleClassName="text-sm mt-1"
                useCardHeader={false}
                className="mb-4"
              />
              <ul className="space-y-2">
                <BulletListItem title="Sandboxed execution" color="chart-3" variant="dot" showPlate={false} />
                <BulletListItem title="Cascading shutdown" color="chart-3" variant="dot" showPlate={false} />
                <BulletListItem title="VRAM cleanup" color="chart-3" variant="dot" showPlate={false} />
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <IconCardHeader
                icon={Shield}
                iconTone="chart-2"
                iconSize="sm"
                title="Zero-Trust Architecture"
                subtitle="Defense-in-depth with timing-safe auth and audit logs."
                titleClassName="text-lg"
                subtitleClassName="text-sm mt-1"
                useCardHeader={false}
                className="mb-4"
              />
              <ul className="space-y-2">
                <BulletListItem title="Timing-safe authentication" color="chart-2" variant="dot" showPlate={false} />
                <BulletListItem title="Immutable audit logs" color="chart-2" variant="dot" showPlate={false} />
                <BulletListItem title="Input validation" color="chart-2" variant="dot" showPlate={false} />
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </SectionContainer>
  )
}
