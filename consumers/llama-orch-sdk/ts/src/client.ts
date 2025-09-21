// Design-phase stub client — signatures only; all methods throw.
// Mirrors consumers/llama-orch-sdk/.docs/03-client.md

import type {
  EngineCapability,
  TaskRequest,
  AdmissionResponse,
  SessionInfo,
  SSEEvent,
} from "./types";

export interface ClientOptions {
  baseURL?: string;
  apiKey?: string; // sent as X-API-Key when provided
  timeoutMs?: number; // for non-streaming calls
}

export class OrchestratorClient {
  private readonly baseURL: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;

  constructor(opts: ClientOptions = {}) {
    this.baseURL = opts.baseURL ?? "http://127.0.0.1:8080/";
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? 30_000;
  }

  async list_engines(): Promise<EngineCapability[]> {
    throw new Error("unimplemented");
  }

  async enqueue_task(req: TaskRequest): Promise<AdmissionResponse> {
    void req;
    throw new Error("unimplemented");
  }

  async *stream_task(taskId: string): AsyncIterable<SSEEvent> {
    void taskId;
    throw new Error("unimplemented");
  }

  async cancel_task(taskId: string): Promise<void> {
    void taskId;
    throw new Error("unimplemented");
  }

  async get_session(sessionId: string): Promise<SessionInfo> {
    void sessionId;
    throw new Error("unimplemented");
  }
}
