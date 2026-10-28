// A fake ComfyUI server for tests: the HTTP endpoints this package calls plus
// a minimal websocket (RFC 6455 handshake and unmasked server→client text
// frames), so the client's websocket path and its polling fallback can both
// be exercised without a real server.

import { createHash, randomUUID } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo, Socket } from "node:net";

import { createImage, encodePng } from "../../src/image.js";
import type { ApiWorkflow, HistoryEntry, ObjectInfo } from "../../src/types.js";
import { makeObjectInfo } from "./object-info.js";

export type Scenario =
  | { kind: "success" }
  | { kind: "validation-400"; body: Record<string, unknown> }
  | { kind: "execution-error"; nodeId: string; nodeType: string; message: string }
  | { kind: "no-output" }
  | { kind: "never-finish" }
  | { kind: "view-404" };

export type FakeComfyOptions = {
  objectInfo?: ObjectInfo;
  /** Accept websocket upgrades (default true). */
  websocket?: boolean;
  /** Execution time: from execution_start to completion. */
  runDelayMs?: number;
  /**
   * Time each prompt waits in the queue before execution starts (as behind
   * other clients' prompts on a shared server); GET /queue lists it as pending.
   */
  queueDelayMs?: number | ((index: number) => number);
  /** Decide the outcome for each queued prompt. */
  scenario?: (graph: ApiWorkflow, index: number) => Scenario;
  /** Produce the output PNG for a successful prompt. */
  render?: (graph: ApiWorkflow, index: number) => Buffer;
  /** Also record failures in /history (as ComfyUI does). Default true. */
  historyOnError?: boolean;
};

export type FakeComfy = {
  url: string;
  prompts: { id: string; graph: ApiWorkflow; clientId: string }[];
  uploads: { name: string; bytes: Buffer }[];
  interrupts: unknown[];
  queueDeletes: unknown[];
  requests: string[];
  close: () => Promise<void>;
};

export function solidPng(width: number, height: number, rgb: [number, number, number] = [200, 50, 50]): Buffer {
  return encodePng(createImage(width, height, { r: rgb[0], g: rgb[1], b: rgb[2] }));
}

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

function wsFrame(text: string): Buffer {
  const payload = Buffer.from(text, "utf8");
  if (payload.length < 126) return Buffer.concat([Buffer.from([0x81, payload.length]), payload]);
  if (payload.length < 65536) {
    const h = Buffer.alloc(4);
    h[0] = 0x81;
    h[1] = 126;
    h.writeUInt16BE(payload.length, 2);
    return Buffer.concat([h, payload]);
  }
  const h = Buffer.alloc(10);
  h[0] = 0x81;
  h[1] = 127;
  h.writeBigUInt64BE(BigInt(payload.length), 2);
  return Buffer.concat([h, payload]);
}

/** Minimal multipart parser: returns the `image` part's filename and bytes. */
function parseMultipartImage(body: Buffer, contentType: string): { filename: string; bytes: Buffer } | null {
  const m = /boundary=(?:"([^"]+)"|([^;]+))/.exec(contentType);
  if (!m) return null;
  const boundary = Buffer.from(`--${m[1] ?? m[2]}`);
  let start = body.indexOf(boundary);
  while (start >= 0) {
    const next = body.indexOf(boundary, start + boundary.length);
    if (next < 0) break;
    const part = body.subarray(start + boundary.length + 2, next - 2);
    const headerEnd = part.indexOf("\r\n\r\n");
    const headers = part.subarray(0, headerEnd).toString("utf8");
    if (/name="image"/.test(headers)) {
      const fn = /filename="([^"]+)"/.exec(headers)?.[1] ?? "upload.png";
      return { filename: fn, bytes: Buffer.from(part.subarray(headerEnd + 4)) };
    }
    start = next;
  }
  return null;
}

export async function startFakeComfy(options: FakeComfyOptions = {}): Promise<FakeComfy> {
  const objectInfo = options.objectInfo ?? makeObjectInfo();
  const history = new Map<string, HistoryEntry>();
  const outputs = new Map<string, Buffer>();
  const sockets = new Map<string, Socket>();
  const allSockets = new Set<Socket>();
  const timers = new Set<NodeJS.Timeout>();
  const state: FakeComfy = { url: "", prompts: [], uploads: [], interrupts: [], queueDeletes: [], requests: [], close: async () => undefined };
  const delay = options.runDelayMs ?? 30;

  const send = (clientId: string, type: string, data: Record<string, unknown>) => {
    const s = sockets.get(clientId);
    if (s && !s.destroyed) s.write(wsFrame(JSON.stringify({ type, data })));
  };

  const schedule = (fn: () => void, ms: number) => {
    const t = setTimeout(() => {
      timers.delete(t);
      fn();
    }, ms);
    timers.add(t);
  };

  const pending = new Set<string>();
  const running = new Set<string>();
  const startedAt = new Map<string, number>();
  const queueDelay = (index: number) => (typeof options.queueDelayMs === "function" ? options.queueDelayMs(index) : options.queueDelayMs ?? 0);

  const execute = (promptId: string, graph: ApiWorkflow, clientId: string, index: number) => {
    pending.add(promptId);
    const wait = queueDelay(index);
    if (wait > 0) schedule(() => start(promptId, graph, clientId, index), wait);
    else start(promptId, graph, clientId, index);
  };

  const start = (promptId: string, graph: ApiWorkflow, clientId: string, index: number) => {
    pending.delete(promptId);
    running.add(promptId);
    startedAt.set(promptId, Date.now());
    const scenario = options.scenario?.(graph, index) ?? { kind: "success" };
    send(clientId, "execution_start", { prompt_id: promptId });
    if (scenario.kind === "never-finish") return;
    schedule(() => {
      running.delete(promptId);
      const saveIds = Object.entries(graph).filter(([, n]) => n.class_type === "SaveImage" || n.class_type === "PreviewImage").map(([id]) => id);
      const samplerId = Object.entries(graph).find(([, n]) => n.class_type.startsWith("KSampler"))?.[0] ?? "x";
      send(clientId, "executing", { node: samplerId, prompt_id: promptId });
      send(clientId, "progress", { value: 1, max: 2, node: samplerId, prompt_id: promptId });
      send(clientId, "progress", { value: 2, max: 2, node: samplerId, prompt_id: promptId });
      if (scenario.kind === "execution-error") {
        const data = {
          prompt_id: promptId,
          node_id: scenario.nodeId,
          node_type: scenario.nodeType,
          exception_message: scenario.message,
          exception_type: "RuntimeError",
          traceback: [],
        };
        if (options.historyOnError !== false) {
          history.set(promptId, { outputs: {}, status: { status_str: "error", completed: false, messages: [["execution_error", data]] } });
        }
        send(clientId, "execution_error", data);
        return;
      }
      const entryOutputs: HistoryEntry["outputs"] = {};
      if (scenario.kind !== "no-output") {
        for (const sid of saveIds) {
          const filename = `${String(graph[sid]!.inputs.filename_prefix ?? "ComfyUI").replace(/\//g, "_")}_${index}_${sid}.png`;
          const bytes = options.render?.(graph, index) ?? solidPng(64, 64);
          if (scenario.kind !== "view-404") outputs.set(filename, bytes);
          entryOutputs[sid] = { images: [{ filename, subfolder: "", type: graph[sid]!.class_type === "SaveImage" ? "output" : "temp" }] };
        }
      }
      history.set(promptId, {
        outputs: entryOutputs,
        status: {
          status_str: "success",
          completed: true,
          messages: [
            ["execution_start", { prompt_id: promptId, timestamp: startedAt.get(promptId) }],
            ["execution_success", { prompt_id: promptId, timestamp: Date.now() }],
          ],
        },
      });
      send(clientId, "execution_success", { prompt_id: promptId });
      send(clientId, "executing", { node: null, prompt_id: promptId });
    }, delay);
  };

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://x");
    state.requests.push(`${req.method} ${url.pathname}`);
    try {
      if (req.method === "GET" && url.pathname === "/object_info") return sendJson(res, 200, objectInfo);
      if (req.method === "GET" && url.pathname === "/system_stats") {
        return sendJson(res, 200, { system: { os: "test", comfyui_version: "0.0.0-test" }, devices: [{ name: "fake", type: "cpu" }] });
      }
      if (req.method === "GET" && url.pathname === "/models") return sendJson(res, 200, ["checkpoints", "loras", "upscale_models"]);
      if (req.method === "GET" && url.pathname.startsWith("/models/")) {
        const folder = decodeURIComponent(url.pathname.slice("/models/".length));
        const map: Record<string, string[]> = {
          checkpoints: (objectInfo.CheckpointLoaderSimple?.input.required?.ckpt_name?.[0] as string[]) ?? [],
          loras: (objectInfo.LoraLoader?.input.required?.lora_name?.[0] as string[]) ?? [],
        };
        return sendJson(res, 200, map[folder] ?? []);
      }
      if (req.method === "POST" && url.pathname === "/prompt") {
        const body = JSON.parse((await readBody(req)).toString("utf8")) as { prompt: ApiWorkflow; client_id: string };
        const index = state.prompts.length;
        const scenario = options.scenario?.(body.prompt, index);
        if (scenario?.kind === "validation-400") {
          state.prompts.push({ id: "rejected", graph: body.prompt, clientId: body.client_id });
          return sendJson(res, 400, scenario.body);
        }
        const id = randomUUID();
        state.prompts.push({ id, graph: body.prompt, clientId: body.client_id });
        sendJson(res, 200, { prompt_id: id, number: index, node_errors: {} });
        execute(id, body.prompt, body.client_id, index);
        return;
      }
      if (req.method === "GET" && url.pathname === "/queue") {
        const item = (id: string) => [0, id, {}, {}, []];
        return sendJson(res, 200, { queue_running: [...running].map(item), queue_pending: [...pending].map(item) });
      }
      if (req.method === "GET" && url.pathname.startsWith("/history/")) {
        const id = decodeURIComponent(url.pathname.slice("/history/".length));
        const entry = history.get(id);
        return sendJson(res, 200, entry ? { [id]: entry } : {});
      }
      if (req.method === "GET" && url.pathname === "/view") {
        const bytes = outputs.get(url.searchParams.get("filename") ?? "");
        if (!bytes) {
          res.writeHead(404);
          res.end("not found");
          return;
        }
        res.writeHead(200, { "content-type": "image/png" });
        res.end(bytes);
        return;
      }
      if (req.method === "POST" && url.pathname === "/upload/image") {
        const part = parseMultipartImage(await readBody(req), String(req.headers["content-type"] ?? ""));
        if (!part) return sendJson(res, 400, { error: "no image" });
        state.uploads.push({ name: part.filename, bytes: part.bytes });
        return sendJson(res, 200, { name: part.filename, subfolder: "", type: "input" });
      }
      if (req.method === "POST" && url.pathname === "/interrupt") {
        state.interrupts.push(JSON.parse((await readBody(req)).toString("utf8") || "{}"));
        res.writeHead(200);
        res.end();
        return;
      }
      if (req.method === "POST" && url.pathname === "/queue") {
        state.queueDeletes.push(JSON.parse((await readBody(req)).toString("utf8") || "{}"));
        res.writeHead(200);
        res.end();
        return;
      }
      res.writeHead(404);
      res.end();
    } catch (err) {
      sendJson(res, 500, { error: String(err) });
    }
  });

  server.on("upgrade", (req: IncomingMessage, socket: Socket) => {
    allSockets.add(socket);
    socket.on("error", () => undefined);
    const url = new URL(req.url ?? "/", "http://x");
    if (options.websocket === false || url.pathname !== "/ws") {
      socket.end("HTTP/1.1 404 Not Found\r\n\r\n");
      return;
    }
    const key = String(req.headers["sec-websocket-key"] ?? "");
    const accept = createHash("sha1").update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`).digest("base64");
    socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
    const clientId = url.searchParams.get("clientId") ?? "";
    sockets.set(clientId, socket);
    socket.write(wsFrame(JSON.stringify({ type: "status", data: { status: { exec_info: { queue_remaining: 0 } }, sid: clientId } })));
    socket.on("data", (chunk: Buffer) => {
      // Client close frame (opcode 0x8): answer and end.
      if (chunk.length > 0 && (chunk[0]! & 0x0f) === 0x8) {
        socket.end(Buffer.from([0x88, 0x00]));
      }
    });
    socket.on("close", () => {
      if (sockets.get(clientId) === socket) sockets.delete(clientId);
      allSockets.delete(socket);
    });
  });
  server.on("connection", (s) => {
    allSockets.add(s);
    s.on("close", () => allSockets.delete(s));
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  state.url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  state.close = async () => {
    for (const t of timers) clearTimeout(t);
    for (const s of allSockets) s.destroy();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  };
  return state;
}
