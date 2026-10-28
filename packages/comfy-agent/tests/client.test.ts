import { createServer } from "node:net";

import { afterEach, describe, expect, it } from "vitest";

import { ComfyClient } from "../src/client.js";
import {
  ComfyConnectionError,
  ComfyExecutionError,
  ComfyInterruptedError,
  ComfyMissingOutputError,
  ComfyOutputFetchError,
  ComfyTimeoutError,
  ComfyValidationError,
  GraphValidationError,
} from "../src/errors.js";
import { buildTxt2Img } from "../src/graphs.js";
import type { ProgressEvent } from "../src/types.js";
import { solidPng, startFakeComfy, type FakeComfy, type FakeComfyOptions } from "./fixtures/fake-comfy.js";

const servers: FakeComfy[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => s.close()));
});

async function setup(opts: FakeComfyOptions = {}, clientOpts: { ws?: boolean } = {}) {
  const fake = await startFakeComfy(opts);
  servers.push(fake);
  const client = new ComfyClient({
    baseUrl: fake.url,
    pollIntervalMs: 20,
    ...(clientOpts.ws === false ? { WebSocket: null } : {}),
  });
  return { fake, client };
}

function graph() {
  return buildTxt2Img({
    checkpoint: "sd_xl_base_1.0.safetensors",
    positive: "a red lantern",
    negative: "blurry",
    width: 768,
    height: 768,
    steps: 10,
    cfg: 6,
    sampler: "euler",
    scheduler: "normal",
    seed: 1,
  }).graph;
}

async function closedPort(): Promise<number> {
  const srv = createServer();
  await new Promise<void>((r) => srv.listen(0, "127.0.0.1", r));
  const port = (srv.address() as { port: number }).port;
  await new Promise<void>((r) => srv.close(() => r()));
  return port;
}

describe("ComfyClient endpoints", () => {
  it("caches /object_info and reads models and system stats", async () => {
    const { fake, client } = await setup();
    const a = await client.getObjectInfo();
    const b = await client.getObjectInfo();
    expect(a).toBe(b);
    expect(fake.requests.filter((r) => r === "GET /object_info")).toHaveLength(1);
    await client.getObjectInfo({ refresh: true });
    expect(fake.requests.filter((r) => r === "GET /object_info")).toHaveLength(2);
    expect(await client.listModels("checkpoints")).toContain("sd_xl_base_1.0.safetensors");
    expect(await client.listModelFolders()).toContain("loras");
    expect((await client.systemStats()).system.comfyui_version).toBe("0.0.0-test");
  });

  it("runs a prompt and observes completion over the websocket", async () => {
    const png = solidPng(32, 32);
    const { fake, client } = await setup({ render: () => png });
    const events: ProgressEvent[] = [];
    const result = await client.run(graph(), { onProgress: (e) => events.push(e), timeoutMs: 5000 });
    expect(result.via).toBe("websocket");
    expect(result.images).toHaveLength(1);
    expect(result.images[0]!.bytes.equals(png)).toBe(true);
    expect(fake.prompts[0]!.clientId).toBe(client.clientId);
    expect(events.some((e) => e.type === "progress" && e.value === 2 && e.max === 2)).toBe(true);
    expect(events[0]!.type).toBe("queued");
  });

  it("falls back to /history polling when the websocket is unavailable", async () => {
    const { client } = await setup({ websocket: false });
    const result = await client.run(graph(), { timeoutMs: 5000 });
    expect(result.via).toBe("polling");
    expect(result.images).toHaveLength(1);
  });

  it("polls when the client is configured without a websocket", async () => {
    const { client } = await setup({}, { ws: false });
    const result = await client.run(graph(), { timeoutMs: 5000 });
    expect(result.via).toBe("polling");
  });

  it("uploads images and sends interrupts and queue deletes", async () => {
    const { fake, client } = await setup();
    const up = await client.uploadImage(solidPng(8, 8), "ref.png");
    expect(up.name).toBe("ref.png");
    expect(fake.uploads[0]!.bytes.equals(solidPng(8, 8))).toBe(true);
    await client.interrupt("abc");
    await client.deleteFromQueue(["abc"]);
    expect(fake.interrupts).toEqual([{ prompt_id: "abc" }]);
    expect(fake.queueDeletes).toEqual([{ delete: ["abc"] }]);
  });

  it("validates against /object_info before queueing", async () => {
    const { fake, client } = await setup();
    const g = graph();
    g.ckpt!.inputs.ckpt_name = "missing.safetensors";
    await expect(client.run(g)).rejects.toBeInstanceOf(GraphValidationError);
    expect(fake.prompts).toHaveLength(0);
  });
});

describe("ComfyClient errors are reported as themselves", () => {
  it("connection refused → ComfyConnectionError", async () => {
    const port = await closedPort();
    const client = new ComfyClient({ baseUrl: `http://127.0.0.1:${port}`, WebSocket: null });
    const err = await client.systemStats().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ComfyConnectionError);
    expect((err as ComfyConnectionError).code).toBe("COMFY_CONNECTION");
    expect((err as Error).message).toContain("ECONNREFUSED");
    const runErr = await client.run(graph(), { validate: false }).catch((e: unknown) => e);
    expect(runErr).toBeInstanceOf(ComfyConnectionError);
  });

  it("HTTP 400 from /prompt → ComfyValidationError with node_errors by node id", async () => {
    const body = {
      error: { type: "prompt_outputs_failed_validation", message: "Prompt outputs failed validation", details: "", extra_info: {} },
      node_errors: {
        sampler: {
          errors: [{ type: "value_not_in_list", message: "Value not in list", details: "sampler_name: 'bogus' not in [...]", extra_info: { input_name: "sampler_name" } }],
          dependent_outputs: ["save"],
          class_type: "KSampler",
        },
      },
    };
    const { client } = await setup({ scenario: () => ({ kind: "validation-400", body }) });
    const err = await client.run(graph(), { validate: false }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ComfyValidationError);
    const v = err as ComfyValidationError;
    expect(v.error.message).toBe("Prompt outputs failed validation");
    expect(Object.keys(v.nodeErrors)).toEqual(["sampler"]);
    expect(v.nodeErrors.sampler!.classType).toBe("KSampler");
    expect(v.nodeErrors.sampler!.errors[0]!.inputName).toBe("sampler_name");
    expect(v.message).toContain("sampler (KSampler)");
  });

  it("websocket execution_error → ComfyExecutionError", async () => {
    const { client } = await setup({
      historyOnError: false,
      scenario: () => ({ kind: "execution-error", nodeId: "sampler", nodeType: "KSampler", message: "MPS backend out of memory" }),
    });
    const err = await client.run(graph(), { timeoutMs: 5000 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ComfyExecutionError);
    const x = err as ComfyExecutionError;
    expect(x.nodeId).toBe("sampler");
    expect(x.nodeType).toBe("KSampler");
    expect(x.exceptionType).toBe("RuntimeError");
    expect(x.message).toContain("MPS backend out of memory");
  });

  it("execution error seen only through /history polling → ComfyExecutionError", async () => {
    const { client } = await setup({
      websocket: false,
      scenario: () => ({ kind: "execution-error", nodeId: "decode", nodeType: "VAEDecode", message: "boom" }),
    });
    const err = await client.run(graph(), { timeoutMs: 5000 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ComfyExecutionError);
    expect((err as ComfyExecutionError).nodeId).toBe("decode");
  });

  it("completed without images → ComfyMissingOutputError", async () => {
    const { client } = await setup({ scenario: () => ({ kind: "no-output" }) });
    const err = await client.run(graph(), { timeoutMs: 5000 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ComfyMissingOutputError);
    expect((err as Error).message).toContain("no outputs");
  });

  it("/view failure → ComfyOutputFetchError, not a timeout", async () => {
    const { client } = await setup({ scenario: () => ({ kind: "view-404" }) });
    const err = await client.run(graph(), { timeoutMs: 5000 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ComfyOutputFetchError);
    expect(err).not.toBeInstanceOf(ComfyTimeoutError);
    expect((err as ComfyOutputFetchError).status).toBe(404);
  });

  it("deadline passes while running → ComfyTimeoutError, and the prompt is cancelled", async () => {
    const { fake, client } = await setup({ scenario: () => ({ kind: "never-finish" }) });
    const err = await client.run(graph(), { timeoutMs: 250 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ComfyTimeoutError);
    expect((err as ComfyTimeoutError).timeoutMs).toBe(250);
    expect((err as ComfyTimeoutError).lastStatus).toBe("running");
    await new Promise((r) => setTimeout(r, 50));
    expect(fake.interrupts.length).toBeGreaterThan(0);
  });

  it("caller abort → ComfyInterruptedError", async () => {
    const { client } = await setup({ scenario: () => ({ kind: "never-finish" }) });
    const ac = new AbortController();
    setTimeout(() => ac.abort(), 100);
    const err = await client.run(graph(), { timeoutMs: 5000, signal: ac.signal }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ComfyInterruptedError);
  });

  it("caller abort also removes the prompt from the queue, so a still-pending prompt never runs", async () => {
    // /interrupt only stops a running prompt; a prompt waiting behind other
    // clients' jobs must be deleted from the queue as well.
    const { fake, client } = await setup({ scenario: () => ({ kind: "never-finish" }) });
    const ac = new AbortController();
    setTimeout(() => ac.abort(), 100);
    await client.run(graph(), { timeoutMs: 5000, signal: ac.signal }).catch(() => undefined);
    await new Promise((r) => setTimeout(r, 50));
    expect(fake.queueDeletes).toEqual([{ delete: [fake.prompts[0]!.id] }]);
  });
});
