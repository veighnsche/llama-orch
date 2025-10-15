import { Badge } from "@rbee/ui/atoms/Badge";
import { Card, CardContent } from "@rbee/ui/atoms/Card";
import { ConsoleOutput } from "@rbee/ui/atoms/ConsoleOutput";
import { Separator } from "@rbee/ui/atoms/Separator";
import { IconCardHeader, IconPlate, SectionContainer } from "@rbee/ui/molecules";
import { cn } from "@rbee/ui/utils";
import { ArrowDown, GitBranch, Network } from "lucide-react";

export function CrossNodeOrchestration() {
  return (
    <SectionContainer
      title="Cross-Pool Orchestration"
      bgVariant="background"
      subtitle="Seamlessly orchestrate AI workloads across your entire network. One command runs inference on any machine in your pool."
    >
      <div className="max-w-6xl mx-auto">
        {/* Overline badge */}
        <div className="flex justify-center mb-8">
          <Badge variant="secondary">Distributed execution</Badge>
        </div>

        <div className="grid gap-8 lg:grid-cols-2 items-start">
          {/* Pool Registry Management Card */}
          <Card className="animate-in fade-in slide-in-from-left-4 duration-500">
            <IconCardHeader
              icon={Network}
              title="Pool Registry Management"
              subtitle="Configure remote machines once. rbee-keeper handles SSH, validates connectivity, and keeps your pool registry synced."
              iconTone="primary"
              iconSize="md"
            />
            <CardContent className="space-y-6">

              <ConsoleOutput
                variant="terminal"
                copyable
                copyText={`# Add a remote machine to your pool\n$ rbee-keeper setup add-node \\\n  --name workstation \\\n  --ssh-host workstation.home.arpa \\\n  --ssh-user vince \\\n  --ssh-key ~/.ssh/id_ed25519\n\n# Run inference on that machine\n$ rbee-keeper infer --node workstation \\\n  --model hf:meta-llama/Llama-3.1-8B \\\n  --prompt "write a short story"`}
                className="font-mono text-sm"
              >
                <div className="text-muted-foreground">
                  # Add a remote machine to your pool
                </div>
                <div className="text-chart-3 mt-2">
                  $ rbee-keeper setup add-node \
                  <br />
                  {"  "}--name workstation \
                  <br />
                  {"  "}--ssh-host workstation.home.arpa \
                  <br />
                  {"  "}--ssh-user vince \
                  <br />
                  {"  "}--ssh-key ~/.ssh/id_ed25519
                </div>
                <div className="text-muted-foreground mt-4">
                  # Run inference on that machine
                </div>
                <div className="text-chart-3 mt-2">
                  $ rbee-keeper infer --node workstation \
                  <br />
                  {"  "}--model hf:meta-llama/Llama-3.1-8B \
                  <br />
                  {"  "}--prompt &quot;write a short story&quot;
                </div>
              </ConsoleOutput>

              <div className="grid sm:grid-cols-3 gap-3">
                <FeatureMiniCard
                  title="Automatic Detection"
                  description="Detects CUDA, Metal, CPU & device counts."
                />
                <FeatureMiniCard
                  title="SSH Validation"
                  description="Connectivity tested before save."
                />
                <FeatureMiniCard
                  title="Zero Config"
                  description="No manual setup on remote nodes."
                />
              </div>
            </CardContent>
          </Card>

          <Separator className="lg:hidden my-2 opacity-40" />

          {/* Automatic Worker Provisioning Card */}
          <Card className="animate-in fade-in slide-in-from-right-4 duration-500 delay-100">
            <CardContent className="space-y-6 pt-6">
              <div className="flex items-start gap-4">
                <IconPlate
                  icon={GitBranch}
                  tone="chart-2"
                  size="md"
                  shape="rounded"
                  className="flex-shrink-0"
                />
                <div>
                  <h3 className="text-2xl font-bold tracking-tight text-foreground mb-2">
                    Automatic Worker Provisioning
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    rbee spawns workers via SSH on demand and shuts them down cleanly. No manual daemons.
                  </p>
                </div>
              </div>

              {/* Diagram */}
              <div className="relative bg-background rounded-xl p-6 overflow-hidden">
                <div className="space-y-4">
                  {/* Row 1: queen-rbee */}
                  <div className="flex items-center gap-3">
                    <div className="flex-1">
                      <DiagramNode name="queen-rbee" label="Orchestrator" tone="primary" />
                    </div>
                  </div>

                  {/* Arrow SSH */}
                  <DiagramArrow label="SSH" indent="pl-8" />

                  {/* Row 2: rbee-hive */}
                  <div className="flex items-center gap-3 pl-8">
                    <div className="flex-1">
                      <DiagramNode name="rbee-hive" label="Pool manager" tone="chart-2" />
                    </div>
                  </div>

                  {/* Arrow Spawns */}
                  <DiagramArrow label="Spawns" indent="pl-16" />

                  {/* Row 3: worker-rbee */}
                  <div className="flex items-center gap-3 pl-16">
                    <div className="flex-1">
                      <DiagramNode name="worker-rbee" label="Inference worker" tone="chart-3" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Legend */}
              <div className="grid sm:grid-cols-3 gap-3 text-sm text-muted-foreground">
                <LegendItem label="On-demand start" />
                <LegendItem label="Clean shutdown" />
                <LegendItem label="No daemon drift" />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </SectionContainer>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper Components
// ──────────────────────────────────────────────────────────────────────────────

interface FeatureMiniCardProps {
  title: string;
  description: string;
}

function FeatureMiniCard({ title, description }: FeatureMiniCardProps) {
  return (
    <div className="bg-background rounded-lg p-4 transition-transform hover:-translate-y-0.5">
      <div className="text-sm font-semibold text-foreground mb-1">{title}</div>
      <div className="text-xs text-muted-foreground">{description}</div>
    </div>
  );
}

interface DiagramNodeProps {
  name: string;
  label: string;
  tone: "primary" | "chart-2" | "chart-3";
}

function DiagramNode({ name, label, tone }: DiagramNodeProps) {
  const toneClasses = {
    primary: "bg-primary/10 border-primary/20",
    "chart-2": "bg-chart-2/10 border-chart-2/20",
    "chart-3": "bg-chart-3/10 border-chart-3/20",
  };

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 border rounded-lg px-4 py-2",
        toneClasses[tone]
      )}
    >
      <span className="font-mono text-sm font-semibold text-foreground">
        {name}
      </span>
      <Badge variant="secondary" className="text-xs">
        {label}
      </Badge>
    </div>
  );
}

interface DiagramArrowProps {
  label: string;
  indent?: string;
}

function DiagramArrow({ label, indent = "" }: DiagramArrowProps) {
  return (
    <div className={cn("flex items-center gap-2", indent)}>
      <ArrowDown className="size-4 text-muted-foreground" aria-hidden="true" />
      <span className="text-xs text-muted-foreground font-medium">{label}</span>
    </div>
  );
}

interface LegendItemProps {
  label: string;
}

function LegendItem({ label }: LegendItemProps) {
  return (
    <div className="flex items-center gap-2">
      <div
        className="size-2 rounded-full bg-chart-3 shrink-0"
        aria-hidden="true"
      />
      <span>{label}</span>
    </div>
  );
}
