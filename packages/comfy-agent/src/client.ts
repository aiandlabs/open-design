import { randomUUID } from "node:crypto";

import {
  ComfyConnectionError,
  ComfyExecutionError,
  ComfyHttpError,
  ComfyInterruptedError,
  ComfyMissingOutputError,
  ComfyOutputFetchError,
  ComfyTimeoutError,
  ComfyValidationError,
  type ComfyNodeError,
} from "./errors.js";
import { assertLocalOnly, assertValidGraph } from "./validate.js";
import type {
  ApiWorkflow,
  HistoryEntry,
  ImageRef,
  ObjectInfo,
  OutputImage,
  ProgressEvent,
  RunResult,
  SystemStats,
} from "./types.js";

type WebSocketCtor = new (url: string) => WebSocket;

export type ComfyClientOptions = {
  baseUrl: string;
  fetch?: typeof fetch;
  /** WebSocket constructor; defaults to the Node 24 global. Pass `null` to force polling. */
  WebSocket?: WebSocketCtor | null;
  clientId?: string;
  /** History poll interval while no websocket is available. */
  pollIntervalMs?: number;
  /** Per-HTTP-request timeout (not the prompt deadline). */
  requestTimeoutMs?: number;
  /** How long to wait for the websocket handshake before falling back to polling. */
  wsConnectTimeoutMs?: number;
};

export type RunOptions = {
  /** Deadline for the prompt to finish, from the moment it is queued. */
  timeoutMs?: number;
  onProgress?: (event: ProgressEvent) => void;
  signal?: AbortSignal;
  /** Validate against /object_info before queueing (default true). */
  validate?: boolean;
  /** Only return images from these node ids (default: every image output). */
  outputNodeIds?: string[];
};

const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000;

export class ComfyClient {
  readonly baseUrl: string;
  readonly clientId: string;
  private readonly fetchImpl: typeof fetch;
  private readonly WebSocketImpl: WebSocketCtor | null;
  private readonly pollIntervalMs: number;
  private readonly requestTimeoutMs: number;
  private readonly wsConnectTimeoutMs: number;
  private objectInfoCache: Promise<ObjectInfo> | null = null;

  constructor(options: ComfyClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.clientId = options.clientId ?? randomUUID();
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
    const globalWs = (globalThis as { WebSocket?: WebSocketCtor }).WebSocket;
    this.WebSocketImpl = options.WebSocket === undefined ? (globalWs ?? null) : options.WebSocket;
    this.pollIntervalMs = options.pollIntervalMs ?? 1000;
    this.requestTimeoutMs = options.requestTimeoutMs ?? 30_000;
    this.wsConnectTimeoutMs = options.wsConnectTimeoutMs ?? 3000;
  }

  url(path: string): string {
    return `${this.baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
  }

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    const url = this.url(path);
    const timeout = AbortSignal.timeout(this.requestTimeoutMs);
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
    try {
      return await this.fetchImpl(url, { ...init, signal });
    } catch (err) {
      if (init.signal?.aborted) throw err;
      throw new ComfyConnectionError(url, timeout.aborted ? new Error(`no response within ${this.requestTimeoutMs} ms`) : err);
    }
  }

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await this.request(path, init);
    const text = await res.text();
    if (!res.ok) throw new ComfyHttpError(this.url(path), res.status, text);
    try {
      return JSON.parse(text) as T;
    } catch {
      throw new ComfyHttpError(this.url(path), res.status, `invalid JSON: ${text.slice(0, 200)}`);
    }
  }

  /** `GET /object_info`, cached for the life of the client. */
  getObjectInfo(options: { refresh?: boolean } = {}): Promise<ObjectInfo> {
    if (!this.objectInfoCache || options.refresh) {
      const pending = this.json<ObjectInfo>("/object_info");
      this.objectInfoCache = pending;
      pending.catch(() => {
        if (this.objectInfoCache === pending) this.objectInfoCache = null;
      });
    }
    return this.objectInfoCache;
  }

  invalidateObjectInfo(): void {
    this.objectInfoCache = null;
  }

  systemStats(): Promise<SystemStats> {
    return this.json<SystemStats>("/system_stats");
  }

  listModelFolders(): Promise<string[]> {
    return this.json<string[]>("/models");
  }

  listModels(folder: string): Promise<string[]> {
    return this.json<string[]>(`/models/${encodeURIComponent(folder)}`);
  }

  getHistory(promptId: string): Promise<HistoryEntry | null> {
    return this.json<Record<string, HistoryEntry>>(`/history/${encodeURIComponent(promptId)}`).then(
      (all) => all[promptId] ?? null,
    );
  }

  /** `POST /prompt`. Maps a 400 to {@link ComfyValidationError} with per-node errors. */
  async queuePrompt(graph: ApiWorkflow, options: { front?: boolean } = {}): Promise<{ promptId: string; number?: number }> {
    const body: Record<string, unknown> = { prompt: graph, client_id: this.clientId };
    if (options.front) body.front = true;
    const res = await this.request("/prompt", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      // handled below
    }
    if (res.status === 400 && parsed && typeof parsed === "object") {
      throw validationErrorFrom(parsed as Record<string, unknown>);
    }
    if (!res.ok) throw new ComfyHttpError(this.url("/prompt"), res.status, text);
    const ok = (parsed ?? {}) as { prompt_id?: string; number?: number; node_errors?: Record<string, unknown> };
    if (!ok.prompt_id) throw new ComfyHttpError(this.url("/prompt"), res.status, `no prompt_id in response: ${text.slice(0, 200)}`);
    if (ok.node_errors && Object.keys(ok.node_errors).length > 0) {
      throw validationErrorFrom(ok as Record<string, unknown>);
    }
    return ok.number === undefined ? { promptId: ok.prompt_id } : { promptId: ok.prompt_id, number: ok.number };
  }

  /** `GET /view`. Failures are {@link ComfyOutputFetchError}, never a timeout. */
  async view(ref: ImageRef): Promise<Buffer> {
    const qs = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder ?? "", type: ref.type ?? "output" });
    let res: Response;
    try {
      res = await this.request(`/view?${qs.toString()}`);
    } catch (err) {
      throw new ComfyOutputFetchError(ref.filename, undefined, err instanceof Error ? err.message : String(err), err);
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new ComfyOutputFetchError(ref.filename, res.status, body.slice(0, 200) || res.statusText);
    }
    const bytes = Buffer.from(await res.arrayBuffer());
    if (bytes.length === 0) throw new ComfyOutputFetchError(ref.filename, res.status, "empty body");
    return bytes;
  }

  /** `POST /upload/image` (multipart). Returns the name to put in `LoadImage.image`. */
  async uploadImage(
    bytes: Uint8Array,
    filename: string,
    options: { overwrite?: boolean; subfolder?: string; type?: "input" | "temp" } = {},
  ): Promise<ImageRef & { name: string }> {
    const form = new FormData();
    form.append("image", new Blob([new Uint8Array(bytes)], { type: "image/png" }), filename);
    form.append("overwrite", options.overwrite === false ? "false" : "true");
    if (options.subfolder) form.append("subfolder", options.subfolder);
    form.append("type", options.type ?? "input");
    const res = await this.request("/upload/image", { method: "POST", body: form });
    const text = await res.text();
    if (!res.ok) throw new ComfyHttpError(this.url("/upload/image"), res.status, text);
    const parsed = JSON.parse(text) as { name: string; subfolder?: string; type?: string };
    const subfolder = parsed.subfolder ?? "";
    return {
      name: subfolder ? `${subfolder}/${parsed.name}` : parsed.name,
      filename: parsed.name,
      subfolder,
      type: parsed.type ?? "input",
    };
  }

  /** `POST /interrupt`. With a prompt id, newer servers only stop that prompt. */
  async interrupt(promptId?: string): Promise<void> {
    const res = await this.request("/interrupt", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(promptId ? { prompt_id: promptId } : {}),
    });
    if (!res.ok) throw new ComfyHttpError(this.url("/interrupt"), res.status, await res.text());
  }

  /**
   * Stop a prompt we no longer want, wherever it is: delete it from the queue
   * (it may still be waiting behind other clients' jobs, which `/interrupt`
   * does not touch) and interrupt it in case it is already running.
   * Best effort; never throws.
   */
  private cancelPrompt(promptId: string): void {
    void this.deleteFromQueue([promptId]).catch(() => undefined);
    void this.interrupt(promptId).catch(() => undefined);
  }

  /** Remove still-pending prompts from the queue. */
  async deleteFromQueue(promptIds: string[]): Promise<void> {
    const res = await this.request("/queue", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ delete: promptIds }),
    });
    if (!res.ok) throw new ComfyHttpError(this.url("/queue"), res.status, await res.text());
  }

  /**
   * Validate, queue, wait for completion, and fetch every image output.
   *
   * Completion is observed over the websocket when it connects, with a slower
   * `/history` poll alongside it in case a message is missed; without a
   * websocket the history poll alone drives completion.
   */
  async run(graph: ApiWorkflow, options: RunOptions = {}): Promise<RunResult> {
    const info = await this.getObjectInfo();
    if (options.validate !== false) assertValidGraph(graph, info);
    else assertLocalOnly(graph, info);
    const started = Date.now();
    const ws = await this.openSocket();
    try {
      const queued = await this.queuePrompt(graph);
      options.onProgress?.({ type: "queued", promptId: queued.promptId, ...(queued.number === undefined ? {} : { number: queued.number }) });
      const { entry, via } = await this.waitFor(queued.promptId, ws, options);
      const images = await this.collectImages(queued.promptId, entry, options.outputNodeIds);
      return { promptId: queued.promptId, images, durationMs: Date.now() - started, via };
    } finally {
      try {
        ws?.close();
      } catch {
        // ignore
      }
    }
  }

  private async openSocket(): Promise<WebSocket | null> {
    if (!this.WebSocketImpl) return null;
    const wsUrl = `${this.baseUrl.replace(/^http/i, "ws")}/ws?clientId=${encodeURIComponent(this.clientId)}`;
    let socket: WebSocket;
    try {
      socket = new this.WebSocketImpl(wsUrl);
    } catch {
      return null;
    }
    return new Promise<WebSocket | null>((resolve) => {
      const timer = setTimeout(() => {
        cleanup();
        try {
          socket.close();
        } catch {
          // ignore
        }
        resolve(null);
      }, this.wsConnectTimeoutMs);
      const onOpen = () => {
        cleanup();
        resolve(socket);
      };
      const onFail = () => {
        cleanup();
        resolve(null);
      };
      const cleanup = () => {
        clearTimeout(timer);
        socket.removeEventListener("open", onOpen);
        socket.removeEventListener("error", onFail);
        socket.removeEventListener("close", onFail);
      };
      socket.addEventListener("open", onOpen);
      socket.addEventListener("error", onFail);
      socket.addEventListener("close", onFail);
    });
  }

  private async waitFor(
    promptId: string,
    ws: WebSocket | null,
    options: RunOptions,
  ): Promise<{ entry: HistoryEntry; via: "websocket" | "polling" }> {
    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const deadline = Date.now() + timeoutMs;
    let lastStatus = "queued";
    let settledBy = null as "websocket" | "polling" | null;
    let failure = null as Error | null;
    let wakeUp = null as (() => void) | null;
    const wake = () => wakeUp?.();

    const onMessage = (event: MessageEvent) => {
      if (typeof event.data !== "string") return; // binary preview frames
      let msg: { type?: string; data?: Record<string, unknown> };
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      const data = msg.data ?? {};
      if (data.prompt_id !== promptId) return;
      switch (msg.type) {
        case "execution_start":
          lastStatus = "running";
          break;
        case "executing":
          if (data.node === null || data.node === undefined) {
            settledBy ??= "websocket";
            wake();
          } else {
            lastStatus = `running node ${String(data.node)}`;
            options.onProgress?.({ type: "executing", promptId, nodeId: String(data.node) });
          }
          break;
        case "progress":
          options.onProgress?.({
            type: "progress",
            promptId,
            value: Number(data.value ?? 0),
            max: Number(data.max ?? 0),
            ...(data.node == null ? {} : { nodeId: String(data.node) }),
          });
          break;
        case "execution_cached":
          options.onProgress?.({ type: "cached", promptId, nodeIds: Array.isArray(data.nodes) ? data.nodes.map(String) : [] });
          break;
        case "execution_success":
          settledBy ??= "websocket";
          wake();
          break;
        case "execution_error":
          failure ??= executionErrorFrom(promptId, data);
          wake();
          break;
        case "execution_interrupted":
          failure ??= new ComfyInterruptedError(promptId, `interrupted at node ${String(data.node_id ?? "?")}`);
          wake();
          break;
        default:
          break;
      }
    };
    ws?.addEventListener("message", onMessage);
    const onAbort = () => {
      failure ??= new ComfyInterruptedError(promptId, "aborted by caller");
      this.cancelPrompt(promptId);
      wake();
    };
    options.signal?.addEventListener("abort", onAbort);
    // A dropped socket mid-run is not a failure; polling carries on.
    let wsAlive = ws !== null;
    const onClose = () => {
      wsAlive = false;
    };
    ws?.addEventListener("close", onClose);

    try {
      for (;;) {
        if (failure) throw failure;
        if (options.signal?.aborted) throw new ComfyInterruptedError(promptId, "aborted by caller");
        // History is authoritative for both success and error.
        const entry = await this.getHistory(promptId);
        if (entry) {
          const status = entry.status;
          if (status?.status_str === "error") {
            throw historyFailure(promptId, entry);
          }
          if (status?.completed || status?.status_str === "success" || (!status && Object.keys(entry.outputs ?? {}).length > 0)) {
            return { entry, via: settledBy ?? "polling" };
          }
        } else if (settledBy) {
          // The socket said done but history is not written yet; poll quickly.
          lastStatus = "finishing";
        }
        const remaining = deadline - Date.now();
        if (remaining <= 0) {
          this.cancelPrompt(promptId);
          throw new ComfyTimeoutError(promptId, timeoutMs, lastStatus);
        }
        const interval = settledBy
          ? 100
          : wsAlive
            ? Math.max(this.pollIntervalMs * 3, 1500)
            : this.pollIntervalMs;
        await new Promise<void>((resolve) => {
          const timer = setTimeout(resolve, Math.min(interval, remaining));
          wakeUp = () => {
            clearTimeout(timer);
            resolve();
          };
        });
        wakeUp = null;
      }
    } finally {
      ws?.removeEventListener("message", onMessage);
      ws?.removeEventListener("close", onClose);
      options.signal?.removeEventListener("abort", onAbort);
    }
  }

  private async collectImages(promptId: string, entry: HistoryEntry, onlyNodes?: string[]): Promise<OutputImage[]> {
    const outputs = entry.outputs ?? {};
    const refs: (ImageRef & { nodeId: string })[] = [];
    for (const [nodeId, out] of Object.entries(outputs)) {
      if (onlyNodes && !onlyNodes.includes(nodeId)) continue;
      for (const img of out.images ?? []) {
        if (img && typeof img.filename === "string") {
          refs.push({ nodeId, filename: img.filename, subfolder: img.subfolder ?? "", type: img.type ?? "output" });
        }
      }
    }
    if (refs.length === 0) {
      const described = Object.entries(outputs).map(([id, o]) => `${id}: [${Object.keys(o).join(", ")}]`).join("; ");
      throw new ComfyMissingOutputError(
        promptId,
        described ? `output nodes reported ${described} but no images${onlyNodes ? ` for nodes ${onlyNodes.join(", ")}` : ""}` : "history has no outputs (the graph may have no SaveImage/PreviewImage node)",
      );
    }
    // Prefer saved outputs over temp previews when both exist.
    const saved = refs.filter((r) => r.type === "output");
    const chosen = saved.length > 0 ? saved : refs;
    const images: OutputImage[] = [];
    for (const ref of chosen) {
      images.push({ ...ref, bytes: await this.view(ref) });
    }
    return images;
  }
}

function validationErrorFrom(body: Record<string, unknown>): ComfyValidationError {
  const errRaw = (body.error ?? {}) as Record<string, unknown>;
  const error = {
    message: typeof errRaw === "string" ? errRaw : String(errRaw.message ?? "prompt validation failed"),
    ...(typeof errRaw.type === "string" ? { type: errRaw.type } : {}),
    ...(typeof errRaw.details === "string" && errRaw.details ? { details: errRaw.details } : {}),
  };
  const nodeErrors: Record<string, ComfyNodeError> = {};
  const rawNodes = (body.node_errors ?? {}) as Record<string, { errors?: unknown[]; class_type?: string }>;
  for (const [id, n] of Object.entries(rawNodes)) {
    nodeErrors[id] = {
      ...(n.class_type ? { classType: n.class_type } : {}),
      errors: (n.errors ?? []).map((e) => {
        const r = (e ?? {}) as Record<string, unknown>;
        const extra = (r.extra_info ?? {}) as Record<string, unknown>;
        return {
          message: String(r.message ?? "error"),
          ...(typeof r.type === "string" ? { type: r.type } : {}),
          ...(typeof r.details === "string" && r.details ? { details: r.details } : {}),
          ...(typeof extra.input_name === "string" ? { inputName: extra.input_name } : {}),
        };
      }),
    };
  }
  return new ComfyValidationError(error, nodeErrors);
}

function executionErrorFrom(promptId: string, data: Record<string, unknown>): ComfyExecutionError {
  return new ComfyExecutionError({
    promptId,
    ...(data.node_id == null ? {} : { nodeId: String(data.node_id) }),
    ...(typeof data.node_type === "string" ? { nodeType: data.node_type } : {}),
    ...(typeof data.exception_type === "string" ? { exceptionType: data.exception_type } : {}),
    exceptionMessage: String(data.exception_message ?? "unknown error").trim(),
  });
}

function historyFailure(promptId: string, entry: HistoryEntry): Error {
  for (const [type, data] of entry.status?.messages ?? []) {
    if (type === "execution_error") return executionErrorFrom(promptId, data);
    if (type === "execution_interrupted") return new ComfyInterruptedError(promptId, `interrupted at node ${String(data.node_id ?? "?")}`);
  }
  return new ComfyExecutionError({ promptId, exceptionMessage: "history reports status error without details" });
}
