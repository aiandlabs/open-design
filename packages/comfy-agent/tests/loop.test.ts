import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { ComfyClient } from "../src/client.js";
import { ComfyExecutionError, ModelSelectionError } from "../src/errors.js";
import { alphaStats, createImage, decodePng, encodePng, type RgbaImage } from "../src/image.js";
import { AiandClient } from "../src/llm.js";
import { generateAsset } from "../src/loop.js";
import { createToolHandler } from "../src/mcp.js";
import { CRITIC_TOOL, normalizeCriticItems, PLAN_TOOL, REPAIR_TOOL } from "../src/prompts.js";
import type { ApiWorkflow } from "../src/types.js";
import { startFakeComfy, type FakeComfy, type FakeComfyOptions } from "./fixtures/fake-comfy.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  await Promise.all(cleanups.splice(0).map((c) => c()));
});

type ToolCallLog = { tool: string; body: Record<string, unknown> };

/** An ai& API stand-in: answers by the forced tool name with scripted JSON. */
function stubLlm(script: { plan: unknown; critic: (n: number, body: Record<string, unknown>) => unknown; repair?: (n: number, body: Record<string, unknown>) => unknown }) {
  const log: ToolCallLog[] = [];
  let critic = 0;
  let repair = 0;
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/models")) {
      return new Response(JSON.stringify({ data: [
        { id: "test/planner", capabilities: ["vision", "tool_calling"], input_per_1m: "3", output_per_1m: "12" },
        { id: "test/critic", capabilities: ["vision", "tool_calling"], input_per_1m: "0.15", output_per_1m: "0.5" },
        { id: "test/text-only", capabilities: ["chat", "tool_calling"], input_per_1m: "1", output_per_1m: "1" },
      ] }));
    }
    const body = JSON.parse(String(init!.body)) as Record<string, unknown>;
    const tool = (body.tool_choice as { function: { name: string } }).function.name;
    log.push({ tool, body });
    let args: unknown;
    if (tool === PLAN_TOOL) args = script.plan;
    else if (tool === CRITIC_TOOL) args = script.critic(critic++, body);
    else if (tool === REPAIR_TOOL) args = script.repair?.(repair++, body) ?? { rationale: "none", edits: [] };
    return new Response(JSON.stringify({
      model: body.model,
      choices: [{ message: { tool_calls: [{ type: "function", function: { name: tool, arguments: JSON.stringify(args) } }] } }],
      usage: { prompt_tokens: 2000, completion_tokens: 300 },
    }));
  }) as typeof fetch;
  return { llm: new AiandClient({ apiKey: "test-key", baseUrl: "https://llm.test/v1", fetch: fetchImpl }), log };
}

async function setup(opts: FakeComfyOptions) {
  const fake: FakeComfy = await startFakeComfy({ runDelayMs: 5, ...opts });
  const workDir = await mkdtemp(path.join(os.tmpdir(), "comfy-agent-loop-"));
  cleanups.push(() => fake.close(), () => rm(workDir, { recursive: true, force: true }));
  return { fake, comfy: new ComfyClient({ baseUrl: fake.url, pollIntervalMs: 10 }), workDir };
}

function lanternPng(w: number, h: number): Buffer {
  return encodePng(createImage(w, h, { r: 230, g: 230, b: 230 }));
}

function discOnGreen(w: number, h: number): RgbaImage {
  const img = createImage(w, h, { r: 18, g: 232, b: 35 });
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (Math.hypot(x - w / 2, y - h / 2) < w / 4) img.data.set([200, 30, 40, 255], (y * w + x) * 4);
    }
  }
  return img;
}

const PLAN = {
  checklist: [
    { id: "count", text: "Exactly one paper lantern is visible" },
    { id: "background", text: "The background is plain white" },
  ],
  positive_prompt: "red paper lantern, white background",
  negative_prompt: "petals",
  steps: 20,
  cfg: 6,
  sampler_name: "euler",
  scheduler: "karras",
};

const positiveOf = (body: Record<string, unknown>) => JSON.stringify(body.messages);

describe("generateAsset", () => {
  it("repairs a failing checklist item with typed edits and stops when everything passes", async () => {
    const { fake, comfy, workDir } = await setup({ render: (g) => lanternPng(Number(g.latent?.inputs.width ?? 768), Number(g.latent?.inputs.height ?? 768)) });
    const { llm, log } = stubLlm({
      plan: PLAN,
      critic: (n) => ({
        items: [
          { id: "count", pass: n > 0, evidence: n > 0 ? "one lantern" : "about 20 lanterns", repair_hint: n > 0 ? "" : "ask for a single lantern" },
          { id: "background", pass: true },
        ],
        overall: n > 0 ? "good" : "too many lanterns",
      }),
      repair: () => ({
        rationale: "state the count explicitly",
        edits: [
          { op: "set_prompt", text: "a single red paper lantern, solo, white background" },
          { op: "set_negative_prompt", text: "multiple lanterns, petals" },
          { op: "set_seed", seed: 123 },
          { op: "set_sampler", sampler_name: "not_a_sampler" },
          { op: "set_steps", text: "" },
        ],
      }),
    });
    const res = await generateAsset(
      { prompt: "a red paper lantern on a white background", width: 512, height: 512, seed: 7 },
      { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, limits: { maxRounds: 3 } },
    );
    expect(res.ok).toBe(true);
    const t = res.trace;
    expect(t.stopReason).toBe("all-pass");
    expect(t.rounds).toHaveLength(2);
    expect(fake.prompts).toHaveLength(2);
    // SDXL request at 512² generates at 768² and is resized to exactly 512².
    expect(t.server.generationSize).toEqual({ width: 768, height: 768 });
    expect(decodePng(res.png)).toMatchObject({ width: 512, height: 512 });

    const [r1, r2] = t.rounds;
    expect(r1!.score).toEqual({ passed: 2, total: 3, ratio: 2 / 3 });
    expect(r1!.checks).toEqual([{ id: "size", pass: true, detail: "512x512 (requested 512x512)" }]);
    expect(r1!.critic!.items[0]).toMatchObject({ id: "count", pass: false, repair_hint: "ask for a single lantern" });
    expect(r1!.repair!.applied).toHaveLength(3);
    expect(r1!.repair!.rejected).toHaveLength(2);
    expect(r1!.repair!.rejected[0]!.error).toMatch(/not_a_sampler/);
    expect(r1!.repair!.rejected[1]!.error).toMatch(/steps must be a number/);
    expect(r1!.settings).toMatchObject({ seed: 7, sampler: "euler", scheduler: "karras", steps: 20, cfg: 6 });

    const g2 = r2!.graph as ApiWorkflow;
    expect(g2.pos!.inputs.text).toBe("a single red paper lantern, solo, white background");
    expect(g2.neg!.inputs.text).toBe("multiple lanterns, petals");
    expect(g2.sampler!.inputs.seed).toBe(123);
    expect(fake.prompts[1]!.graph).toEqual(g2);
    expect(r2!.score.ratio).toBe(1);
    expect(t.best).toMatchObject({ round: 2 });
    expect(res.graph).toEqual(g2);

    // Every round records timing, images and LLM usage; the trace is on disk.
    for (const r of t.rounds) {
      expect(existsSync(r.imagePath!)).toBe(true);
      expect(existsSync(r.rawImagePath!)).toBe(true);
      expect(r.timing.comfyMs).toBeGreaterThanOrEqual(0);
      expect(r.critic!.call!.usage).toEqual({ inputTokens: 2000, outputTokens: 300 });
    }
    expect(log.map((l) => l.tool)).toEqual([PLAN_TOOL, CRITIC_TOOL, REPAIR_TOOL, CRITIC_TOOL]);
    // 2 planner calls at (2000×3 + 300×12)/1e6 and 2 critic calls at (2000×0.15 + 300×0.5)/1e6.
    expect(t.totals.estimatedCostUsd).toBeCloseTo(2 * 0.0096 + 2 * 0.00045, 10);
    expect(t.totals.reportedCostUsd).toBeNull();
    expect(t.totals.inputTokens).toBe(8000);
    const onDisk = JSON.parse(readFileSync(path.join(workDir, "trace.json"), "utf8"));
    expect(onDisk.id).toBe(t.id);
    // The critic got an image.
    expect(positiveOf(log[1]!.body)).toContain("data:image/png;base64,");
  });

  it("makes transparent assets by keying a planner-chosen flat colour", async () => {
    const { fake, comfy, workDir } = await setup({ render: () => encodePng(discOnGreen(768, 768)) });
    const { llm } = stubLlm({
      plan: { ...PLAN, checklist: [{ id: "subject", text: "One red disc" }], key_color: "#00ff00" },
      critic: () => ({ items: [{ id: "subject", pass: true }] }),
    });
    const res = await generateAsset(
      { prompt: "a red disc icon", width: 256, height: 256, transparent: true },
      { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir },
    );
    expect(res.trace.stopReason).toBe("all-pass");
    const r1 = res.trace.rounds[0]!;
    expect(r1.checks.map((c) => [c.id, c.pass])).toEqual([
      ["size", true],
      ["alpha", true],
      ["flat_background", true],
      ["key_residue", true],
    ]);
    expect(r1.keyColor).toMatchObject({ requested: "#00FF00", borderMatch: 1 });
    const out = decodePng(res.png);
    expect([out.width, out.height]).toEqual([256, 256]);
    expect(out.data[3]).toBe(0); // corner transparent
    expect(alphaStats(out).borderTransparent).toBe(1);
    const sent = fake.prompts[0]!.graph;
    expect(String(sent.pos!.inputs.text)).toContain("chroma-key green background");
    expect(String(sent.neg!.inputs.text)).toContain("gradient background");
  });

  it("describes the cut-out view the critic actually receives in both system and user text", async () => {
    const { comfy, workDir } = await setup({ render: () => encodePng(discOnGreen(768, 768)) });
    const { llm, log } = stubLlm({
      plan: { ...PLAN, checklist: [{ id: "subject", text: "One red disc" }], key_color: "#00ff00" },
      critic: () => ({ items: [{ id: "subject", pass: true }] }),
    });
    await generateAsset(
      { prompt: "a red disc icon", width: 256, height: 256, transparent: true },
      { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir },
    );
    const critic = log.find((l) => l.tool === CRITIC_TOOL)!.body;
    const messages = critic.messages as { role: string; content: unknown }[];
    const system = String(messages.find((m) => m.role === "system")!.content);
    const user = JSON.stringify(messages.find((m) => m.role === "user")!.content);
    // The image is cutoutSheet(): white panel and near-black panel. Nothing is shown on grey.
    expect(user).toContain("on white (left) and on near-black (right)");
    expect(system).not.toMatch(/grey|gray/i);
    expect(system).toMatch(/white/);
    expect(system).toMatch(/near-black/);
  });

  it("treats set_seed without a value as a request for a new seed", async () => {
    const { fake, comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm } = stubLlm({
      plan: PLAN,
      critic: (n) => ({ items: [{ id: "count", pass: n > 0 }, { id: "background", pass: true }] }),
      repair: () => ({ rationale: "new composition", edits: [{ op: "set_seed", text: "" }] }),
    });
    const res = await generateAsset({ prompt: "lantern", width: 768, height: 768, seed: 11 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic" });
    expect(res.trace.rounds[0]!.repair!.applied).toHaveLength(1);
    expect(fake.prompts[1]!.graph.sampler!.inputs.seed).not.toBe(11);
  });

  it("switches to img2img from the previous output when the planner asks", async () => {
    const { fake, comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm } = stubLlm({
      plan: PLAN,
      critic: (n) => ({ items: [{ id: "count", pass: n > 0 }, { id: "background", pass: true }] }),
      repair: () => ({ rationale: "keep composition", edits: [{ op: "img2img_from_previous", denoise: 0.5 }] }),
    });
    const res = await generateAsset({ prompt: "lantern", width: 768, height: 768, seed: 1 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic" });
    expect(res.trace.stopReason).toBe("all-pass");
    expect(fake.uploads).toHaveLength(1);
    const g2 = fake.prompts[1]!.graph;
    expect(g2.load!.inputs.image).toBe(fake.uploads[0]!.name);
    expect(g2.sampler!.inputs).toMatchObject({ denoise: 0.5, latent_image: ["encode", 0] });
  });

  it("inpaints a region of the previous output with a generated mask", async () => {
    const { fake, comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm } = stubLlm({
      plan: PLAN,
      critic: (n) => ({ items: [{ id: "count", pass: n > 0 }, { id: "background", pass: true }] }),
      repair: () => ({ rationale: "fix left side", edits: [{ op: "inpaint_previous", x: 0, y: 0.25, width: 0.4, height: 0.5, denoise: 0.8 }] }),
    });
    await generateAsset({ prompt: "lantern", width: 768, height: 768 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic" });
    expect(fake.uploads).toHaveLength(2);
    const mask = decodePng(fake.uploads[1]!.bytes);
    expect(mask.data[(384 * 768 + 100) * 4]).toBe(255);
    expect(mask.data[(384 * 768 + 700) * 4]).toBe(0);
    expect(fake.prompts[1]!.graph.noiseMask).toBeDefined();
  });

  it("stops at the round limit and returns the best round", async () => {
    const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm } = stubLlm({
      plan: PLAN,
      critic: (n) => ({ items: [{ id: "count", pass: n === 0 }, { id: "background", pass: false }] }),
      repair: () => ({ rationale: "try again", edits: [] }),
    });
    const res = await generateAsset({ prompt: "lantern", width: 768, height: 768, seed: 5 }, { comfy, llm, workDir, limits: { maxRounds: 2 }, plannerModel: "test/planner", criticModel: "test/critic" });
    expect(res.ok).toBe(false);
    expect(res.trace.stopReason).toBe("max-rounds");
    expect(res.trace.best!.round).toBe(1);
    // No edits → the loop changed the seed so round 2 was not an identical graph.
    expect(res.trace.rounds[1]!.settings.seed).not.toBe(5);
  });

  it("stops at the cost limit", async () => {
    const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm } = stubLlm({ plan: PLAN, critic: () => ({ items: [{ id: "count", pass: false }, { id: "background", pass: true }] }) });
    const res = await generateAsset({ prompt: "lantern", width: 768, height: 768 }, { comfy, llm, workDir, limits: { maxRounds: 5, maxCostUsd: 0.005 }, plannerModel: "test/planner", criticModel: "test/critic" });
    expect(res.trace.stopReason).toBe("cost-limit");
    expect(res.trace.rounds).toHaveLength(1);
  });

  it("stops at the time limit", async () => {
    const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768), runDelayMs: 60 });
    const { llm } = stubLlm({ plan: PLAN, critic: () => ({ items: [{ id: "count", pass: false }, { id: "background", pass: true }] }) });
    const res = await generateAsset({ prompt: "lantern", width: 768, height: 768 }, { comfy, llm, workDir, limits: { maxRounds: 5, maxTimeMs: 50 }, plannerModel: "test/planner", criticModel: "test/critic" });
    expect(res.trace.stopReason).toBe("time-limit");
    expect(res.trace.rounds).toHaveLength(1);
  });

  it("returns the best image when a later round fails, and reports the failure as itself", async () => {
    const { comfy, workDir } = await setup({
      render: () => lanternPng(768, 768),
      scenario: (_g, i) => (i === 1 ? { kind: "execution-error", nodeId: "sampler", nodeType: "KSampler", message: "out of memory" } : { kind: "success" }),
    });
    const { llm } = stubLlm({
      plan: PLAN,
      critic: () => ({ items: [{ id: "count", pass: false }, { id: "background", pass: true }] }),
      repair: () => ({ rationale: "more steps", edits: [{ op: "set_steps", steps: 30 }] }),
    });
    const res = await generateAsset({ prompt: "lantern", width: 768, height: 768 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic" });
    expect(res.trace.stopReason).toBe("error");
    expect(res.trace.rounds[1]!.error).toMatchObject({ code: "COMFY_EXECUTION" });
    expect(res.trace.best!.round).toBe(1);
  });

  it("propagates a first-round ComfyUI failure", async () => {
    const { comfy, workDir } = await setup({ scenario: () => ({ kind: "execution-error", nodeId: "decode", nodeType: "VAEDecode", message: "bad" }) });
    const { llm } = stubLlm({ plan: PLAN, critic: () => ({ items: [] }) });
    await expect(generateAsset({ prompt: "x", width: 512, height: 512 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic" })).rejects.toBeInstanceOf(ComfyExecutionError);
  });
});

describe("generateAsset model selection", () => {
  it("picks both models from the catalog when none is configured", async () => {
    const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm, log } = stubLlm({ plan: PLAN, critic: () => ({ items: [{ id: "count", pass: true }, { id: "background", pass: true }] }) });
    const res = await generateAsset({ prompt: "lantern", width: 768, height: 768 }, { comfy, llm, workDir });
    // No preferred id is listed by this stub, so the first capable model wins.
    expect(res.trace.models).toMatchObject({ planner: "test/planner", critic: "test/planner" });
    expect(log.every((l) => l.body.model === "test/planner")).toBe(true);
  });

  it("rejects a configured critic without vision before it touches ComfyUI", async () => {
    const { fake, comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm, log } = stubLlm({ plan: PLAN, critic: () => ({ items: [] }) });
    const err = await generateAsset({ prompt: "x", width: 512, height: 512 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/text-only" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ModelSelectionError);
    expect((err as ModelSelectionError).toJSON()).toMatchObject({ code: "LLM_MODEL", role: "critic", reason: "missing-capability", missing: ["vision"] });
    expect(log).toHaveLength(0);
    expect(fake.requests).toEqual([]);
  });
});

describe("critic output normalisation", () => {
  it("accepts a flattened single item, string booleans and JSON-string items", () => {
    expect(normalizeCriticItems({ id: "count", pass: "false", evidence: "nine lanterns", repair_hint: "one lantern" }))
      .toEqual([{ id: "count", pass: false, evidence: "nine lanterns", repair_hint: "one lantern" }]);
    expect(normalizeCriticItems({ items: JSON.stringify([{ id: "a", pass: "pass" }]) })).toEqual([{ id: "a", pass: true }]);
    expect(normalizeCriticItems({ items: { a: true, b: { pass: "no", reason: "r" } } })).toEqual([{ id: "a", pass: true }, { id: "b", pass: false, evidence: "r" }]);
    expect(normalizeCriticItems({ items: [{ id: "a", pass: "maybe" }] })).toEqual([]);
    expect(normalizeCriticItems("nope")).toEqual([]);
  });

  it("retries the critic once when grades are missing, and keeps the raw reply in the trace", async () => {
    const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm, log } = stubLlm({
      plan: PLAN,
      critic: (n) => (n === 0
        ? { id: "count", pass: "false", evidence: "nine lanterns" }
        : { items: [{ id: "count", pass: true }, { id: "background", pass: true }] }),
    });
    const res = await generateAsset({ prompt: "lantern", width: 768, height: 768 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic" });
    expect(log.filter((l) => l.tool === CRITIC_TOOL)).toHaveLength(2);
    const critic = res.trace.rounds[0]!.critic!;
    expect(critic.raw).toEqual({ id: "count", pass: "false", evidence: "nine lanterns" });
    expect(critic.retries).toHaveLength(1);
    expect(critic.items.map((i) => i.pass)).toEqual([true, true]);
    expect(res.trace.stopReason).toBe("all-pass");
  });
});

describe("MCP tool handler", () => {
  it("inspects the server, runs workflows, generates assets and returns traces", async () => {
    const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
    const { llm } = stubLlm({ plan: PLAN, critic: () => ({ items: [{ id: "count", pass: true }, { id: "background", pass: true }] }) });
    const handle = createToolHandler({
      config: { comfyUrl: comfy.baseUrl, aiandBaseUrl: "https://llm.test/v1", plannerModel: "test/planner", criticModel: "test/critic", apiShape: "openai" },
      outDir: workDir,
      comfy,
      llm,
    });
    const inspect = JSON.parse(((await handle("comfy_inspect", {})).content[0] as { text: string }).text);
    expect(inspect.capabilities.checkpoints).toContain("sd_xl_base_1.0.safetensors");
    expect(inspect.baseGraphs).toEqual(["txt2img", "img2img", "inpaint"]);

    const run = await handle("comfy_run_workflow", { prompt: "a lantern", width: 512, height: 512, seed: 3 });
    const runOut = JSON.parse((run.content[0] as { text: string }).text);
    expect(existsSync(runOut.path)).toBe(true);
    expect(runOut.settings.seed).toBe(3);
    expect(run.content[1]!.type).toBe("image");

    const gen = await handle("comfy_generate_asset", { prompt: "a lantern", width: 512, height: 512, max_rounds: 1 });
    const genOut = JSON.parse((gen.content[0] as { text: string }).text);
    expect(genOut.ok).toBe(true);
    const trace = JSON.parse(((await handle("comfy_get_trace", { trace_id: genOut.traceId })).content[0] as { text: string }).text);
    expect(trace.id).toBe(genOut.traceId);
    expect(trace.rounds).toHaveLength(1);

    const bad = await handle("comfy_run_workflow", { workflow: { nodes: [], definitions: { subgraphs: [{}] } } });
    expect(bad.isError).toBe(true);
    expect((bad.content[0] as { text: string }).text).toContain("WORKFLOW_FORMAT");
  });
});
