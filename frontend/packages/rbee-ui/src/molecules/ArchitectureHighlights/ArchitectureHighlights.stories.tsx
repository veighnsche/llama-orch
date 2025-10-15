import type { Meta, StoryObj } from "@storybook/react";
import { ArchitectureHighlights } from "./ArchitectureHighlights";
import type { ArchitectureHighlight } from "./ArchitectureHighlights";

const meta = {
  title: "Molecules/ArchitectureHighlights",
  component: ArchitectureHighlights,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof ArchitectureHighlights>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultHighlights: ArchitectureHighlight[] = [
  {
    title: "BDD-Driven Development",
    details: ["42/62 scenarios passing (68% complete)", "Live CI coverage"],
  },
  {
    title: "Cascading Shutdown Guarantee",
    details: ["No orphaned processes. Clean VRAM lifecycle."],
  },
  {
    title: "Process Isolation",
    details: ["Worker-level sandboxes. Zero cross-leak."],
  },
  {
    title: "Protocol-Aware Orchestration",
    details: ["SSE, JSON, binary protocols."],
  },
  {
    title: "Smart/Dumb Separation",
    details: ["Central brain, distributed execution."],
  },
];

export const Default: Story = {
  args: {
    highlights: defaultHighlights,
  },
};

export const SingleDetail: Story = {
  args: {
    highlights: [
      {
        title: "Zero-Copy Architecture",
        details: ["Direct memory access for maximum throughput."],
      },
      {
        title: "Async Runtime",
        details: ["Tokio-based concurrency."],
      },
      {
        title: "Type Safety",
        details: ["Compile-time guarantees."],
      },
    ],
  },
};

export const MultipleDetails: Story = {
  args: {
    highlights: [
      {
        title: "Security First",
        details: [
          "End-to-end encryption",
          "Zero-trust architecture",
          "Regular security audits",
          "Compliance with SOC 2 Type II",
        ],
      },
      {
        title: "High Availability",
        details: [
          "99.99% uptime SLA",
          "Multi-region deployment",
          "Automatic failover",
        ],
      },
    ],
  },
};

export const MinimalHighlights: Story = {
  args: {
    highlights: [
      {
        title: "Fast",
        details: ["Sub-millisecond latency."],
      },
      {
        title: "Reliable",
        details: ["Battle-tested in production."],
      },
    ],
  },
};

export const DevelopmentFeatures: Story = {
  args: {
    highlights: [
      {
        title: "Hot Module Replacement",
        details: ["Instant feedback during development", "No full page reloads"],
      },
      {
        title: "TypeScript Support",
        details: ["Full type inference", "IDE autocomplete"],
      },
      {
        title: "Testing Built-in",
        details: ["Unit tests with Vitest", "E2E with Playwright"],
      },
      {
        title: "Developer Experience",
        details: ["Clear error messages", "Comprehensive documentation"],
      },
    ],
  },
};

export const PerformanceMetrics: Story = {
  args: {
    highlights: [
      {
        title: "Throughput",
        details: ["10,000 requests/second", "Linear scaling"],
      },
      {
        title: "Latency",
        details: ["P50: 5ms", "P99: 25ms", "P99.9: 100ms"],
      },
      {
        title: "Resource Usage",
        details: ["50MB memory footprint", "< 1% CPU idle"],
      },
    ],
  },
};
