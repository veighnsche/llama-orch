import { Hono } from "hono";

const app = new Hono<{ Bindings: CloudflareBindings }>();

// List all available worker variants
app.get("/workers", (c) => {
  return c.json({
    workers: [
      {
        id: "llm-worker-rbee-cpu",
        variant: "cpu",
        description: "LLM worker for rbee system (CPU-only)",
        arch: ["x86_64", "aarch64"],
        pkgbuild_url: "/workers/cpu/PKGBUILD"
      },
      {
        id: "llm-worker-rbee-cuda",
        variant: "cuda",
        description: "LLM worker for rbee system (NVIDIA CUDA)",
        arch: ["x86_64"],
        pkgbuild_url: "/workers/cuda/PKGBUILD"
      },
      {
        id: "llm-worker-rbee-metal",
        variant: "metal",
        description: "LLM worker for rbee system (Apple Metal)",
        arch: ["aarch64"],
        pkgbuild_url: "/workers/metal/PKGBUILD"
      }
    ]
  });
});

// Serve PKGBUILD files
app.get("/workers/cpu/PKGBUILD", async (c) => {
  const pkgbuild = await c.env.ASSETS.fetch(new Request("http://placeholder/pkgbuilds/llm-worker-cpu.PKGBUILD"));
  return new Response(pkgbuild.body, {
    headers: { "Content-Type": "text/plain" }
  });
});

app.get("/workers/cuda/PKGBUILD", async (c) => {
  const pkgbuild = await c.env.ASSETS.fetch(new Request("http://placeholder/pkgbuilds/llm-worker-cuda.PKGBUILD"));
  return new Response(pkgbuild.body, {
    headers: { "Content-Type": "text/plain" }
  });
});

app.get("/workers/metal/PKGBUILD", async (c) => {
  const pkgbuild = await c.env.ASSETS.fetch(new Request("http://placeholder/pkgbuilds/llm-worker-metal.PKGBUILD"));
  return new Response(pkgbuild.body, {
    headers: { "Content-Type": "text/plain" }
  });
});

export default app;
