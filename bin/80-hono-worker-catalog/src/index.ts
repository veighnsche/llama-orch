// Worker Catalog Service
// Provides metadata and PKGBUILD files for worker installation

import { Hono } from "hono";
import { cors } from "hono/cors";
import { routes } from "./routes";

const app = new Hono<{ Bindings: CloudflareBindings }>();

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CORS MIDDLEWARE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Allow requests from hive UI and other local services

app.use("/*", cors({
  origin: [
    "http://localhost:7836",  // Hive UI
    "http://localhost:8500",  // Queen Rbee
    "http://localhost:8501",  // Rbee Keeper
    "http://127.0.0.1:7836",
    "http://127.0.0.1:8500",
    "http://127.0.0.1:8501",
  ],
  allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowHeaders: ["Content-Type", "Authorization"],
  exposeHeaders: ["Content-Length"],
  maxAge: 600,
  credentials: true,
}));

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ROUTES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.route("/", routes);

// Health check
app.get("/health", (c) => {
  return c.json({ 
    status: "ok",
    service: "worker-catalog",
    version: "0.1.0"
  });
});

export default app;
