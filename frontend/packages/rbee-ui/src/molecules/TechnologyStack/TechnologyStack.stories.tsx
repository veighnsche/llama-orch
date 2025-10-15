import type { Meta, StoryObj } from "@storybook/react";
import { TechnologyStack } from "./TechnologyStack";
import type { TechItem } from "./TechnologyStack";

const meta = {
  title: "Molecules/TechnologyStack",
  component: TechnologyStack,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof TechnologyStack>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultTechnologies: TechItem[] = [
  {
    name: "Rust",
    description: "Performance + memory safety.",
    ariaLabel: "Tech: Rust",
  },
  {
    name: "Candle ML",
    description: "Rust-native inference.",
    ariaLabel: "Tech: Candle ML",
  },
  {
    name: "Rhai Scripting",
    description: "Embedded, sandboxed policies.",
    ariaLabel: "Tech: Rhai Scripting",
  },
  {
    name: "SQLite",
    description: "Embedded, zero-ops DB.",
    ariaLabel: "Tech: SQLite",
  },
  {
    name: "Axum + Vue.js",
    description: "Async backend + modern UI.",
    ariaLabel: "Tech: Axum + Vue.js",
  },
];

export const Default: Story = {
  args: {
    technologies: defaultTechnologies,
    showOpenSourceCTA: true,
    githubUrl: "https://github.com/yourusername/rbee",
    license: "MIT License",
    showArchitectureLink: true,
    architectureUrl: "/docs/architecture",
  },
};

export const WithoutOpenSourceCTA: Story = {
  args: {
    technologies: defaultTechnologies,
    showOpenSourceCTA: false,
    showArchitectureLink: true,
    architectureUrl: "/docs/architecture",
  },
};

export const WithoutArchitectureLink: Story = {
  args: {
    technologies: defaultTechnologies,
    showOpenSourceCTA: true,
    githubUrl: "https://github.com/yourusername/rbee",
    license: "MIT License",
    showArchitectureLink: false,
  },
};

export const MinimalStack: Story = {
  args: {
    technologies: [
      {
        name: "Python",
        description: "Rapid prototyping.",
        ariaLabel: "Tech: Python",
      },
      {
        name: "FastAPI",
        description: "Modern async framework.",
        ariaLabel: "Tech: FastAPI",
      },
    ],
    showOpenSourceCTA: false,
    showArchitectureLink: false,
  },
};

export const CustomLicense: Story = {
  args: {
    technologies: defaultTechnologies,
    showOpenSourceCTA: true,
    githubUrl: "https://github.com/yourusername/project",
    license: "GPL-3.0-or-later",
    showArchitectureLink: true,
    architectureUrl: "/docs/architecture",
  },
};

export const LargeTechStack: Story = {
  args: {
    technologies: [
      {
        name: "Rust",
        description: "Performance + memory safety.",
        ariaLabel: "Tech: Rust",
      },
      {
        name: "TypeScript",
        description: "Type-safe JavaScript.",
        ariaLabel: "Tech: TypeScript",
      },
      {
        name: "React",
        description: "Component-based UI.",
        ariaLabel: "Tech: React",
      },
      {
        name: "Next.js",
        description: "Full-stack React framework.",
        ariaLabel: "Tech: Next.js",
      },
      {
        name: "Tailwind CSS",
        description: "Utility-first styling.",
        ariaLabel: "Tech: Tailwind CSS",
      },
      {
        name: "PostgreSQL",
        description: "Robust relational database.",
        ariaLabel: "Tech: PostgreSQL",
      },
      {
        name: "Redis",
        description: "In-memory caching.",
        ariaLabel: "Tech: Redis",
      },
      {
        name: "Docker",
        description: "Container orchestration.",
        ariaLabel: "Tech: Docker",
      },
    ],
    showOpenSourceCTA: true,
    githubUrl: "https://github.com/yourusername/rbee",
    license: "MIT License",
    showArchitectureLink: true,
    architectureUrl: "/docs/architecture",
  },
};
