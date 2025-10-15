import { GitHubIcon } from "@rbee/ui/atoms";
import {
  CTASection,
  DevelopersCodeExamples,
  DevelopersFeatures,
  DevelopersHero,
  DevelopersSolution,
  DevelopersUseCases,
  EmailCapture,
  PricingSection,
  TestimonialsSection,
} from "@rbee/ui/organisms";
import { ArrowRight } from "lucide-react";
import { ProblemSection } from "@rbee/ui/organisms/ProblemSection";
import { AlertTriangle, DollarSign, Lock } from "lucide-react";
import { HowItWorksSection } from "@rbee/ui/organisms";

export default function DevelopersPage() {
  return (
    <main className="min-h-screen bg-slate-950">
      <DevelopersHero />
      <EmailCapture />
      <ProblemSection
        id="risk"
        kicker="The Hidden Cost of Dependency"
        title="The Hidden Risk of AI-Assisted Development"
        subtitle="You're building complex codebases with AI assistance. But what happens when your provider changes the rules?"
        items={[
          {
            title: "The Model Changes",
            body: "Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your team is blocked.",
            icon: <AlertTriangle className="h-6 w-6" />,
            tone: "destructive",
            tag: "High risk",
          },
          {
            title: "The Price Increases",
            body: "$20/month becomes $200/month. Multiply by your team size. Your AI infrastructure costs spiral out of control.",
            icon: <DollarSign className="h-6 w-6" />,
            tone: "primary",
            tag: "Cost increase: 10x",
          },
          {
            title: "The Provider Shuts Down",
            body: "API deprecated. Service discontinued. Your complex codebase—built with AI assistance—becomes unmaintainable overnight.",
            icon: <Lock className="h-6 w-6" />,
            tone: "destructive",
            tag: "Critical failure",
          },
        ]}
        ctaPrimary={{ label: "Take Control", href: "/getting-started" }}
        ctaSecondary={{ label: "View Documentation", href: "/docs" }}
        ctaCopy="Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external providers."
      />
      <DevelopersSolution />
      <HowItWorksSection
        id="quickstart"
        title="From zero to AI infrastructure in 15 minutes"
        steps={[
          {
            label: "Install rbee",
            block: {
              kind: "terminal",
              title: "terminal",
              lines: (
                <>
                  <div>curl -sSL https://rbee.dev/install.sh | sh</div>
                  <div className="text-slate-400">rbee-keeper daemon start</div>
                </>
              ),
              copyText:
                "curl -sSL https://rbee.dev/install.sh | sh\nrbee-keeper daemon start",
            },
          },
          {
            label: "Add your machines",
            block: {
              kind: "terminal",
              title: "terminal",
              lines: (
                <>
                  <div>
                    rbee-keeper setup add-node --name workstation --ssh-host
                    192.168.1.10
                  </div>
                  <div className="text-slate-400">
                    rbee-keeper setup add-node --name mac --ssh-host
                    192.168.1.20
                  </div>
                </>
              ),
              copyText:
                "rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10\nrbee-keeper setup add-node --name mac --ssh-host 192.168.1.20",
            },
          },
          {
            label: "Configure your IDE",
            block: {
              kind: "terminal",
              title: "terminal",
              lines: (
                <>
                  <div>
                    <span className="text-blue-400">export</span>{" "}
                    OPENAI_API_BASE=http://localhost:8080/v1
                  </div>
                  <div className="text-slate-400">
                    # Now Zed, Cursor, or any OpenAI-compatible tool works!
                  </div>
                </>
              ),
              copyText: "export OPENAI_API_BASE=http://localhost:8080/v1",
            },
          },
          {
            label: "Build AI agents",
            block: {
              kind: "code",
              title: "TypeScript",
              language: "ts",
              lines: (
                <>
                  <div>
                    <span className="text-purple-400">import</span> {"{"} invoke{" "}
                    {"}"} <span className="text-purple-400">from</span>{" "}
                    <span className="text-amber-400">
                      &apos;@rbee/utils&apos;
                    </span>
                    ;
                  </div>
                  <div className="mt-2">
                    <span className="text-blue-400">const</span> code ={" "}
                    <span className="text-blue-400">await</span>{" "}
                    <span className="text-green-400">invoke</span>
                    {"({"}
                  </div>
                  <div className="pl-4">
                    prompt:{" "}
                    <span className="text-amber-400">
                      &apos;Generate API from schema&apos;
                    </span>
                    ,
                  </div>
                  <div className="pl-4">
                    model:{" "}
                    <span className="text-amber-400">
                      &apos;llama-3.1-70b&apos;
                    </span>
                  </div>
                  <div>{"});"}</div>
                </>
              ),
              copyText:
                "import { invoke } from '@llama-orch/utils';\n\nconst code = await invoke({\n  prompt: 'Generate API from schema',\n  model: 'llama-3.1-70b'\n});",
            },
          },
        ]}
      />
      <DevelopersFeatures />
      <DevelopersUseCases />
      <DevelopersCodeExamples />
      <PricingSection
        variant="home"
        showKicker={false}
        showEditorialImage={false}
      />
      <TestimonialsSection
        title="Trusted by Developers Who Value Independence"
        testimonials={[
          {
            avatar: "👨‍💻",
            author: "Alex K.",
            role: "Solo Developer",
            quote:
              "Spent $80/mo on Claude. Now I run Llama-70B on my gaming PC + old workstation. Same quality, $0 cost.",
          },
          {
            avatar: "👩‍💼",
            author: "Sarah M.",
            role: "CTO",
            quote:
              "We pooled our team's hardware and cut AI spend from $500/mo to zero. OpenAI-compatible API—no code changes.",
          },
          {
            avatar: "👨‍🔧",
            author: "Marcus T.",
            role: "DevOps Engineer",
            quote:
              "Cascading shutdown ends orphaned processes and VRAM leaks. Ctrl+C and everything cleans up.",
          },
        ]}
        stats={[
          { value: "1,200+", label: "GitHub stars" },
          { value: "500+", label: "Active installations" },
          { value: "8,000+", label: "GPUs orchestrated" },
          { value: "€0", label: "Avg. monthly cost", tone: "primary" },
        ]}
      />
      <CTASection
        title="Stop Depending on AI Providers. Start Building Today."
        subtitle="Join 500+ developers who've taken control of their AI infrastructure."
        primary={{
          label: "Get Started Free",
          href: "/getting-started",
          iconRight: ArrowRight,
        }}
        secondary={{
          label: "View Documentation",
          href: "/docs",
          iconLeft: GitHubIcon,
          variant: "outline",
        }}
        note="100% open source. No credit card required. Install in 15 minutes."
      />
    </main>
  );
}
