"use client";

import { Badge } from "@rbee/ui/atoms/Badge";
import {
  PlaybookHeader,
  PlaybookItem,
  SectionContainer,
  StatusKPI,
  TerminalWindow,
} from "@rbee/ui/molecules";
import {
  Activity,
  AlertTriangle,
  Check,
  CheckCircle2,
  Database,
  Network,
} from "lucide-react";
import { useCallback } from "react";

export function ErrorHandling() {
  const handleExpandAll = useCallback(() => {
    document
      .querySelectorAll<HTMLDetailsElement>("#playbook details")
      .forEach((d) => (d.open = true));
  }, []);

  const handleCollapseAll = useCallback(() => {
    document
      .querySelectorAll<HTMLDetailsElement>("#playbook details")
      .forEach((d) => (d.open = false));
  }, []);

  return (
    <SectionContainer
      title="Comprehensive Error Handling"
      bgVariant="background"
      subtitle="19+ error scenarios with clear messages and actionable fixes—no cryptic failures."
      eyebrow={<Badge variant="secondary">Resiliency</Badge>}
    >
      <div className="max-w-6xl mx-auto space-y-8">
        {/* 1. Status header (KPIs strip) */}
        <div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2">
          <StatusKPI
            icon={CheckCircle2}
            color="chart-3"
            label="Scenarios covered"
            value="19+"
          />
          <StatusKPI
            icon={Network}
            color="primary"
            label="Auto-retries"
            value="SSH • HTTP • DL"
          />
          <StatusKPI
            icon={Database}
            color="chart-2"
            label="Fail-fast"
            value="Clear suggestions"
          />
        </div>

        {/* 2. Live timeline console */}
        <div className="animate-in fade-in slide-in-from-bottom-2 delay-100">
          <TerminalWindow
            title="error timeline — retries & jitter"
            ariaLabel="Error timeline with retry examples"
            footer={
              <div className="flex items-center gap-2 text-sm text-chart-3">
                <Check className="h-4 w-4" />
                <span>
                  Exponential backoff with random jitter (0.5–1.5×). SSH: 3
                  attempts · HTTP: 3 · Downloads: 6 with resume.
                </span>
              </div>
            }
          >
            <div role="log" aria-live="polite">
              <div className="text-destructive">
                [ssh] attempt 1 → timeout (5000ms)
              </div>
              <div className="text-muted-foreground">
                retry in 0.8× backoff (1.2s jitter)
              </div>
              <div className="text-destructive mt-2">
                [ssh] attempt 2 → auth failed
              </div>
              <div className="text-muted-foreground">
                suggestion: check ~/.ssh/config or key permissions
              </div>
              <div className="text-primary mt-2">
                [http] attempt 1 → 502 Bad Gateway
              </div>
              <div className="text-muted-foreground">
                retry in 1.4× backoff (2.8s jitter)
              </div>
              <div className="text-chart-3 mt-2">
                [download] resumed from 43% — OK
              </div>
              <div className="text-chart-3">
                [worker] graceful shutdown after 30s timeout — OK
              </div>
            </div>
          </TerminalWindow>
        </div>

        {/* 3. Playbook accordion */}
        <div
          id="playbook"
          className="rounded-2xl border border-border bg-card overflow-hidden"
        >
          <PlaybookHeader
            title="Playbook"
            description="19+ scenarios · 4 categories"
            filterCategories={["Network", "Resource", "Model", "Process"]}
            selectedCategories={[]}
            onFilterToggle={() => {}}
            onExpandAll={handleExpandAll}
            onCollapseAll={handleCollapseAll}
          />

          <PlaybookItem
            icon={Network}
            color="warning"
            title="Network & Connectivity"
            checkCount={4}
            severityDots={["destructive", "primary", "chart-3"]}
            description="Detects timeouts, auth failures, and HTTP errors. Retries with exponential backoff + jitter."
            checks={[
              {
                severity: "destructive",
                title: "SSH connection timeout",
                meaning: "Remote host didn't respond within threshold; likely egress or firewall.",
                actionLabel: "Retry with jitter",
                href: "#retry",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "destructive",
                title: "SSH authentication failure",
                meaning: "Keys or agents rejected by server.",
                actionLabel: "Open fix steps",
                href: "#fix",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "primary",
                title: "HTTP connection failures",
                meaning: "Auto-retry on transient TCP resets.",
                actionLabel: "Enable auto-retry",
                href: "#enable",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "chart-3",
                title: "Connection loss during inference",
                meaning: "Stream dropped; partial results were saved.",
                actionLabel: "Resume stream",
                href: "#resume",
                guideLabel: "Timeline",
                guideHref: "#timeline",
              },
            ]}
            footer={
              <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground/80">
                <span>
                  View logs:{" "}
                  <code className="px-1 py-0.5 rounded bg-muted text-foreground/90">
                    ~/.rbee/logs/keeper.log
                  </code>
                </span>
                <a
                  href="#error-timeline"
                  className="underline hover:no-underline"
                >
                  See timeline example
                </a>
              </div>
            }
          />

          <PlaybookItem
            icon={AlertTriangle}
            color="primary"
            title="Resource Errors"
            checkCount={4}
            severityDots={["primary", "destructive"]}
            description="Fail-fast on RAM/VRAM limits with actionable fixes and pre-download disk checks."
            checks={[
              {
                severity: "primary",
                title: "Insufficient RAM",
                meaning: "Process exceeds available memory; swap thrashing.",
                actionLabel: "Lower batch / increase memory",
                href: "#memory",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "destructive",
                title: "VRAM exhausted",
                meaning: "GPU out of memory; no CPU fallback configured.",
                actionLabel: "Use smaller precision",
                href: "#precision",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "primary",
                title: "Disk space pre-check",
                meaning: "Download blocked to prevent low-disk failures.",
                actionLabel: "Choose cache path",
                href: "#cache",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "destructive",
                title: "OOM during model load",
                meaning: "Abort safely before corrupted state.",
                actionLabel: "Stream weights",
                href: "#stream",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
            ]}
            footer={
              <div className="mt-4 text-xs text-muted-foreground/80">
                Docs:{" "}
                <a
                  href="/docs/errors#resource"
                  className="underline hover:no-underline"
                >
                  Resource errors
                </a>
              </div>
            }
          />

          <PlaybookItem
            icon={Database}
            color="chart-2"
            title="Model & Backend"
            checkCount={4}
            severityDots={["chart-2", "primary"]}
            description="Validates model presence, credentials, and backend availability before work starts."
            checks={[
              {
                severity: "chart-2",
                title: "Model 404 (Hugging Face link)",
                meaning: "Requested model not found or renamed.",
                actionLabel: "Select available model",
                href: "#select",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "primary",
                title: "Private model 403",
                meaning: "Token lacks permission for repository.",
                actionLabel: "Fix token scope",
                href: "#token",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "primary",
                title: "Download failures (resume support)",
                meaning: "Interrupted download with resumable chunks.",
                actionLabel: "Resume now",
                href: "#resume",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "chart-2",
                title: "Backend not available",
                meaning: "Health probe failing; show alternatives.",
                actionLabel: "Switch endpoint",
                href: "#switch",
                guideLabel: "Alternatives",
                guideHref: "#alternatives",
              },
            ]}
            footer={
              <div className="mt-4 text-xs text-muted-foreground/80">
                Docs:{" "}
                <a
                  href="/docs/errors#model"
                  className="underline hover:no-underline"
                >
                  Model & backend errors
                </a>
              </div>
            }
          />

          <PlaybookItem
            icon={Activity}
            color="chart-3"
            title="Process Lifecycle"
            checkCount={4}
            severityDots={["chart-3", "destructive"]}
            description="Observes workers from startup to shutdown; offers safe teardown and timeouts."
            checks={[
              {
                severity: "destructive",
                title: "Worker binary missing",
                meaning: "Install steps incomplete; worker cannot spawn.",
                actionLabel: "Run installer",
                href: "#install",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "destructive",
                title: "Crash during startup",
                meaning: "Read early log pointers for root cause.",
                actionLabel: "Open logs",
                href: "#logs",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "chart-3",
                title: "Graceful shutdown",
                meaning: "Drain active requests before exit.",
                actionLabel: "Send SIGTERM",
                href: "#sigterm",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
              {
                severity: "chart-3",
                title: "Force-kill after 30s",
                meaning: "Timeout guard to prevent hung exits.",
                actionLabel: "Adjust timeout",
                href: "#timeout",
                guideLabel: "Guide",
                guideHref: "#guide",
              },
            ]}
            footer={
              <div className="mt-4 text-xs text-muted-foreground/80">
                Docs:{" "}
                <a
                  href="/docs/errors#lifecycle"
                  className="underline hover:no-underline"
                >
                  Process lifecycle errors
                </a>
              </div>
            }
          />
        </div>
      </div>
    </SectionContainer>
  );
}
