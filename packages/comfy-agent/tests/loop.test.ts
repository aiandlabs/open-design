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
import { CRITIC_TOOL, MATCH_TOOL, normalizeCriticItems, PLAN_TOOL, PLANNER_SYSTEM, RECOGNISE_TOOL, REPAIR_TOOL } from "../src/prompts.js";
import type { ApiWorkflow } from "../src/types.js";
import { startFakeComfy, type FakeComfy, type FakeComfyOptions } from "./fixtures/fake-comfy.js";
import { FAMILY_NODES } from "./fixtures/object-info-families.js";
import { MATTING_NODES } from "./fixtures/object-info-matting.js";
import { makeObjectInfo } from "./fixtures/object-info.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  await Promise.all(cleanups.splice(0).map((c) => c()));
});

type ToolCallLog = { tool: string; body: Record<string, unknown> };

/** An ai& API stand-in: answers by the forced tool name with scripted JSON. */
function stubLlm(script: {
  plan: unknown;
  critic: (n: number, body: Record<string, unknown>) => unknown;
  repair?: (n: number, body: Record<string, unknown>) => unknown;
  recognise?: (n: number, body: Record<string, unknown>) => unknown;
  match?: (n: number, body: Record<string, unknown>) => unknown;
}) {
  const log: ToolCallLog[] = [];
  let critic = 0;
  let repair = 0;
  let recognise = 0;
  let match = 0;
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
    else if (tool === RECOGNISE_TOOL) args = script.recognise?.(recognise++, body) ?? { name: "the requested subject" };
    else if (tool === MATCH_TOOL) args = script.match?.(match++, body) ?? { match: true, reason: "same thing" };
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

  describe("with a matting model installed", () => {
    const mattingInfo = () => ({ ...makeObjectInfo(), ...structuredClone(MATTING_NODES) });
    /** What ComfyUI saves after JoinImageWithAlpha: the render's RGB with the matte as alpha. */
    const mattedDisc = (w: number, h: number) => {
      const img = createImage(w, h, { r: 250, g: 250, b: 250, a: 0 });
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) if (Math.hypot(x - w / 2, y - h / 2) < w / 4) img.data.set([200, 30, 40, 255], (y * w + x) * 4);
      }
      return img;
    };

    it("renders on a plain backdrop and cuts the subject out with the matting model", async () => {
      const { fake, comfy, workDir } = await setup({ objectInfo: mattingInfo(), render: () => encodePng(mattedDisc(768, 768)) });
      const { llm, log } = stubLlm({
        plan: { ...PLAN, checklist: [{ id: "subject", text: "One red disc" }], key_color: "#ffffff" },
        critic: () => ({ items: [{ id: "subject", pass: true }] }),
      });
      const res = await generateAsset(
        { prompt: "a red disc icon", width: 256, height: 256, transparent: true },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir },
      );
      expect(res.trace.stopReason).toBe("all-pass");
      expect(res.trace.server.cutout).toEqual({ method: "matte", model: "birefnet.safetensors" });
      const r1 = res.trace.rounds[0]!;
      expect(r1.checks.map((c) => [c.id, c.pass])).toEqual([
        ["size", true],
        ["alpha", true],
        ["matte_residue", true],
      ]);
      expect(r1.cutout).toMatchObject({ method: "matte", backdrop: { requested: "#FFFFFF" } });
      const sent = fake.prompts[0]!.graph;
      expect(Object.values(sent).map((n) => n.class_type)).toEqual(expect.arrayContaining(["RemoveBackground", "InvertMask", "JoinImageWithAlpha"]));
      expect(String(sent.pos!.inputs.text)).toMatch(/plain flat solid white background/);
      // The planner is offered backdrops for matting, not chroma-key colours.
      const planText = positiveOf(log.find((l) => l.tool === PLAN_TOOL)!.body);
      expect(planText).toContain("matting model");
      expect(planText).toContain("#808080");
      expect(planText).not.toContain("chroma-key green");
      // The round's raw image is the opaque render (for edits); the asset is the cut-out.
      expect(alphaStats(decodePng(readFileSync(path.join(workDir, "round-1-raw.png")))).transparent).toBe(0);
      const out = decodePng(res.png);
      expect(out.data[3]).toBe(0);
      expect(alphaStats(out).borderTransparent).toBe(1);
    });

    it("keys with a chroma colour instead when the request asks for it", async () => {
      const { fake, comfy, workDir } = await setup({ objectInfo: mattingInfo(), render: () => encodePng(discOnGreen(768, 768)) });
      const { llm } = stubLlm({
        plan: { ...PLAN, checklist: [{ id: "subject", text: "One red disc" }], key_color: "#00ff00" },
        critic: () => ({ items: [{ id: "subject", pass: true }] }),
      });
      const res = await generateAsset(
        { prompt: "a red disc icon", width: 256, height: 256, transparent: true, cutout: "chroma-key" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir },
      );
      expect(res.trace.server.cutout).toEqual({ method: "chroma-key" });
      expect(res.trace.rounds[0]!.checks.map((c) => c.id)).toEqual(["size", "alpha", "flat_background", "key_residue"]);
      expect(Object.values(fake.prompts[0]!.graph).some((n) => n.class_type === "RemoveBackground")).toBe(false);
    });

    it("falls back to the default backdrop when the planner picks a chroma colour for a matte", async () => {
      const { fake, comfy, workDir } = await setup({ objectInfo: mattingInfo(), render: () => encodePng(mattedDisc(768, 768)) });
      const { llm } = stubLlm({
        plan: { ...PLAN, checklist: [{ id: "subject", text: "One red disc" }], key_color: "#00ff00" },
        critic: () => ({ items: [{ id: "subject", pass: true }] }),
      });
      const res = await generateAsset(
        { prompt: "a red disc icon", width: 256, height: 256, transparent: true },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir },
      );
      expect(res.trace.plan.keyColor).toBe("#FFFFFF");
      expect(String(fake.prompts[0]!.graph.pos!.inputs.text)).not.toContain("green");
    });
  });

  describe("blind recognition", () => {
    const SUBJECT_PLAN = { ...PLAN, subject: "onigiri (Japanese rice ball)" };

    it("fails a round a viewer would not recognise, without telling the viewer the subject, and repairs from what they saw", async () => {
      const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
      const { llm, log } = stubLlm({
        plan: SUBJECT_PLAN,
        critic: () => ({ items: [{ id: "count", pass: true }, { id: "background", pass: true }] }),
        recognise: (n) => (n === 0 ? { name: "a bar of soap", could_also_be: ["a candle"] } : { name: "an onigiri rice ball" }),
        match: (n) => (n === 0 ? { match: false, reason: "soap is not food" } : { match: true, reason: "same" }),
        repair: () => ({ rationale: "restore nori", edits: [{ op: "set_prompt", text: "an onigiri with dark nori" }] }),
      });
      const res = await generateAsset({ prompt: "an onigiri icon, pastel", width: 768, height: 768, seed: 1 }, { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir });
      const r1 = res.trace.rounds[0]!;
      expect(r1.checks.find((c) => c.id === "recognisable")).toMatchObject({ pass: false });
      expect(r1.checks.find((c) => c.id === "recognisable")!.detail).toContain("a bar of soap");
      // The viewer never sees the subject or the prompt.
      const blind = JSON.stringify(log.find((l) => l.tool === RECOGNISE_TOOL)!.body);
      expect(blind).not.toMatch(/onigiri|rice ball|pastel/i);
      expect(JSON.stringify(log.find((l) => l.tool === REPAIR_TOOL)!.body)).toContain("a bar of soap");
      expect(res.trace.stopReason).toBe("all-pass");
      expect(res.trace.rounds[1]!.checks.find((c) => c.id === "recognisable")).toMatchObject({ pass: true });
    });

    it("can be turned off", async () => {
      const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
      const { llm, log } = stubLlm({ plan: SUBJECT_PLAN, critic: () => ({ items: [{ id: "count", pass: true }, { id: "background", pass: true }] }) });
      const res = await generateAsset({ prompt: "an onigiri icon", width: 768, height: 768, seed: 1 }, { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, recognition: false });
      expect(log.some((l) => l.tool === RECOGNISE_TOOL)).toBe(false);
      expect(res.trace.rounds[0]!.checks.some((c) => c.id === "recognisable")).toBe(false);
    });
  });

  describe("icon references", () => {
    const kleinInfo = () => ({ ...makeObjectInfo(), ...structuredClone(FAMILY_NODES), ...structuredClone(MATTING_NODES) });
    /** A matted subject: a coloured disc, optionally with a notch, on a transparent canvas. */
    const subject = (size: number, rgb: [number, number, number], notch = false) => {
      const img = createImage(size, size, { r: 255, g: 255, b: 255, a: 0 });
      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          const inside = Math.hypot(x - size / 2, y - size / 2) < size / 3 && !(notch && x > size / 2 && y < size / 2);
          if (inside) img.data.set(Math.abs(y - size / 2) < size / 20 ? [20, 20, 20, 255] : [...rgb, 255], (y * size + x) * 4);
        }
      }
      return img;
    };
    const CUP_REF = subject(512, [120, 70, 30], true);
    function fakeLibrary() {
      const loads: string[] = [];
      return {
        loads,
        library: {
          indexDigests: { emoji: "e".repeat(64) },
          search: async () => [
            { id: "emoji:2615", kind: "emoji" as const, label: "hot beverage", tags: ["coffee"], score: 4.5 },
            { id: "icon:coffee", kind: "icon" as const, label: "coffee", tags: ["cup"], score: 3 },
          ],
          load: async (id: string, size = 512) => {
            loads.push(`${id}@${size}`);
            return { id, kind: id.startsWith("icon") ? ("icon" as const) : ("emoji" as const), label: id, source: `test source for ${id}`, url: `https://refs.test/${id}`, png: encodePng(id === "emoji:2615" ? CUP_REF : subject(size, [0, 0, 0])) };
          },
        },
      };
    }
    const ICON_PLAN = { ...PLAN, checklist: [{ id: "subject", text: "One coffee cup" }], key_color: "#ffffff" };

    it("conditions an icon on the reference the planner chose, steering away from its style, and records the source", async () => {
      const { fake, comfy, workDir } = await setup({ objectInfo: kleinInfo(), render: () => encodePng(subject(768, [200, 40, 40])) });
      const { library } = fakeLibrary();
      const { llm, log } = stubLlm({
        plan: { ...ICON_PLAN, references: [{ id: "emoji:2615", why: "same concept: a cup with steam" }] },
        critic: () => ({ items: [{ id: "subject", pass: true }] }),
      });
      const res = await generateAsset(
        { prompt: "a coffee cup icon with steam, flat style", width: 512, height: 512, transparent: true, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, references: { library } },
      );
      // The planner saw the candidates as a picture and as a list.
      const planBody = log.find((l) => l.tool === PLAN_TOOL)!.body;
      expect(positiveOf(planBody)).toContain("data:image/png;base64,");
      expect(positiveOf(planBody)).toContain("emoji:2615 hot beverage");
      expect(res.trace.references!.shape).toEqual([
        { id: "emoji:2615", label: "emoji:2615", source: "test source for emoji:2615", url: "https://refs.test/emoji:2615", why: "same concept: a cup with steam" },
      ]);
      expect(res.trace.references!.candidates.map((c) => c.id)).toEqual(["emoji:2615", "icon:coffee"]);
      const sent = fake.prompts[0]!.graph;
      expect(sent.ref1Load!.inputs.image).toBe(fake.uploads[0]!.name);
      expect(sent.ref2Load).toBeUndefined();
      expect(String(sent.pos!.inputs.text)).toMatch(/Image 1 .*shape and concept only/);
      expect(String(sent.pos!.inputs.text)).toMatch(/do not copy/i);
      const r1 = res.trace.rounds[0]!;
      expect(r1.checks.find((c) => c.id === "reference_copy")).toMatchObject({ pass: true });
      expect(r1.referenceSimilarity).toEqual([expect.objectContaining({ id: "emoji:2615", nearCopy: false })]);
      expect(res.trace.stopReason).toBe("all-pass");
    });

    it("makes the planner answer the reference question whenever candidates are listed", async () => {
      const { comfy, workDir } = await setup({ objectInfo: kleinInfo(), render: () => encodePng(subject(768, [200, 40, 40])) });
      const { library } = fakeLibrary();
      const { llm, log } = stubLlm({ plan: { ...ICON_PLAN, references: [] , reference_note: "none fits" }, critic: () => ({ items: [{ id: "subject", pass: true }] }) });
      await generateAsset(
        { prompt: "a coffee cup icon with steam, flat style", width: 512, height: 512, transparent: true, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, references: { library } },
      );
      const body = JSON.stringify(log.find((l) => l.tool === PLAN_TOOL)!.body);
      // The tool schema sent with candidates requires an answer about references.
      expect(body).toMatch(/"required":\["checklist","positive_prompt","negative_prompt","subject","references"\]/);
      expect(JSON.stringify(log.find((l) => l.tool === PLAN_TOOL)!.body)).toMatch(/When a candidate shows the requested subject, choose it/);
    });

    it.each([
      ["the sheet number", "1"],
      ["the sheet number as a number", 1],
      ["the id without its prefix", "2615"],
      ["the id in another case", "EMOJI:2615"],
    ])("accepts a reference given as %s", async (_label, id) => {
      const { comfy, workDir } = await setup({ objectInfo: kleinInfo(), render: () => encodePng(subject(768, [200, 40, 40])) });
      const { library } = fakeLibrary();
      const { llm } = stubLlm({ plan: { ...ICON_PLAN, references: [{ id, why: "cup" }] }, critic: () => ({ items: [{ id: "subject", pass: true }] }) });
      const res = await generateAsset(
        { prompt: "a coffee cup icon with steam, flat style", width: 512, height: 512, transparent: true, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, references: { library } },
      );
      expect(res.trace.references!.shape.map((s) => s.id)).toEqual(["emoji:2615"]);
      expect(res.trace.references!.answer).toEqual([{ id, why: "cup" }]);
    });

    it("says so when the planner's reference matches no candidate, instead of claiming it chose none", async () => {
      const { comfy, workDir } = await setup({ objectInfo: kleinInfo(), render: () => encodePng(subject(768, [200, 40, 40])) });
      const { library } = fakeLibrary();
      const { llm } = stubLlm({ plan: { ...ICON_PLAN, references: [{ id: "teapot", why: "?" }] }, critic: () => ({ items: [{ id: "subject", pass: true }] }) });
      const res = await generateAsset(
        { prompt: "a coffee cup icon with steam, flat style", width: 512, height: 512, transparent: true, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, references: { library } },
      );
      expect(res.trace.references!.shape).toEqual([]);
      expect(res.trace.references!.note).toMatch(/matched no candidate: teapot/);
    });

    it("tells the planner its own checklist items must not contradict the request or each other", () => {
      expect(PLANNER_SYSTEM).toMatch(/must not contradict the request or each other/);
      expect(PLANNER_SYSTEM).toMatch(/keep their real colours/);
      expect(PLANNER_SYSTEM).not.toMatch(/in the requested palette \("a soft muted-green nori wrap"\)/);
    });

    it("does not require a reference answer when no candidates are listed", async () => {
      const { comfy, workDir } = await setup({ objectInfo: kleinInfo(), render: () => encodePng(subject(768, [200, 40, 40])) });
      const { llm, log } = stubLlm({ plan: ICON_PLAN, critic: () => ({ items: [{ id: "subject", pass: true }] }) });
      await generateAsset(
        { prompt: "a coffee cup icon with steam, flat style", width: 512, height: 512, transparent: true, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir },
      );
      const body = JSON.stringify(log.find((l) => l.tool === PLAN_TOOL)!.body);
      expect(body).toMatch(/"required":\["checklist","positive_prompt","negative_prompt","subject"\]/);
    });

    it("fails an output that copies its reference, drops the reference on repair, and does not return the copy", async () => {
      // Round 1 reproduces the reference (recoloured); round 2, without it, draws something new.
      const { fake, comfy, workDir } = await setup({
        objectInfo: kleinInfo(),
        render: (_g, i) => encodePng(i === 0 ? subject(768, [200, 40, 40], true) : subject(768, [200, 40, 40])),
      });
      const { library } = fakeLibrary();
      const { llm, log } = stubLlm({
        plan: { ...ICON_PLAN, references: [{ id: "emoji:2615", why: "cup" }] },
        critic: () => ({ items: [{ id: "subject", pass: true }] }),
        repair: () => ({ rationale: "it copies the reference", edits: [{ op: "drop_reference", id: "emoji:2615" }] }),
      });
      const res = await generateAsset(
        { prompt: "a coffee cup icon", width: 512, height: 512, transparent: true, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, references: { library } },
      );
      const r1 = res.trace.rounds[0]!;
      expect(r1.checks.find((c) => c.id === "reference_copy")).toMatchObject({ pass: false });
      expect(r1.referenceSimilarity![0]!.nearCopy).toBe(true);
      expect(positiveOf(log.find((l) => l.tool === REPAIR_TOOL)!.body)).toContain("reference_copy");
      expect(r1.repair!.applied).toEqual([{ op: "drop_reference", id: "emoji:2615" }]);
      expect(Object.keys(fake.prompts[1]!.graph).some((k) => k.startsWith("ref"))).toBe(false);
      expect(res.trace.best!.round).toBe(2);
      expect(res.trace.stopReason).toBe("all-pass");
    });

    it("never returns a near-copy as the best image while another round is not one", async () => {
      const { comfy, workDir } = await setup({
        objectInfo: kleinInfo(),
        render: (_g, i) => encodePng(i === 0 ? subject(768, [200, 40, 40]) : subject(768, [200, 40, 40], true)),
      });
      const { library } = fakeLibrary();
      const { llm } = stubLlm({
        plan: { ...ICON_PLAN, references: [{ id: "emoji:2615", why: "cup" }] },
        // Round 1 misses a checklist item; round 2 passes it but copies the reference.
        critic: (n) => ({ items: [{ id: "subject", pass: n > 0 }] }),
        repair: () => ({ rationale: "new seed", edits: [{ op: "set_seed", seed: 99 }] }),
      });
      const res = await generateAsset(
        { prompt: "a coffee cup icon", width: 512, height: 512, transparent: true, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, references: { library }, limits: { maxRounds: 2 } },
      );
      expect(res.trace.rounds[1]!.checks.find((c) => c.id === "reference_copy")!.pass).toBe(false);
      expect(res.trace.best!.round).toBe(1);
      expect(res.ok).toBe(false);
    });

    it("uses icons already made for the set as style references and grades consistency against them", async () => {
      const { fake, comfy, workDir } = await setup({ objectInfo: kleinInfo(), render: () => encodePng(subject(768, [200, 40, 40])) });
      const { llm, log } = stubLlm({
        plan: ICON_PLAN,
        critic: () => ({ items: [{ id: "subject", pass: true }, { id: "set_consistency", pass: true }] }),
      });
      const set = [
        { id: "icons/leaf.png", png: encodePng(subject(256, [40, 160, 60])), label: "leaf" },
        { id: "icons/rocket.png", png: encodePng(subject(256, [40, 60, 200])), label: "rocket" },
      ];
      const res = await generateAsset(
        { prompt: "a coffee cup icon", width: 512, height: 512, transparent: true, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, references: { set } },
      );
      expect(res.trace.references!.set).toEqual([{ id: "icons/leaf.png", label: "leaf" }, { id: "icons/rocket.png", label: "rocket" }]);
      expect(res.trace.plan.checklist.map((c) => c.id)).toEqual(["subject", "set_consistency"]);
      const sent = fake.prompts[0]!.graph;
      expect([sent.ref1Load, sent.ref2Load].map((n) => n!.inputs.image)).toEqual(fake.uploads.map((u) => u.name));
      expect(String(sent.pos!.inputs.text)).toMatch(/Images 1–2 are icons already made for the same set/);
      // The critic sees the asset and a sheet of the set.
      const criticBody = log.find((l) => l.tool === CRITIC_TOOL)!.body;
      expect(positiveOf(criticBody).match(/data:image\/png;base64,/g)).toHaveLength(2);
      expect(positiveOf(criticBody)).toContain("same set");
      expect(res.trace.rounds[0]!.referenceSimilarity!.map((s) => s.id)).toEqual(["icons/leaf.png", "icons/rocket.png"]);
    });

    it("generates without references, and says so, on a model that cannot take them", async () => {
      const { fake, comfy, workDir } = await setup({ render: () => lanternPng(1024, 1024) });
      const { library, loads } = fakeLibrary();
      const { llm } = stubLlm({ plan: { ...PLAN, references: [{ id: "emoji:2615", why: "cup" }] }, critic: () => ({ items: [{ id: "count", pass: true }, { id: "background", pass: true }] }) });
      const res = await generateAsset(
        { prompt: "a coffee cup icon", width: 512, height: 512, kind: "icon" },
        { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir, references: { library, set: [{ id: "a.png", png: lanternPng(64, 64) }] } },
      );
      expect(loads).toEqual([]);
      expect(res.trace.references).toMatchObject({ shape: [], set: [], note: expect.stringMatching(/cannot take reference images/) });
      expect(Object.keys(fake.prompts[0]!.graph).some((k) => k.startsWith("ref"))).toBe(false);
    });
  });

  it("records chroma keying as the cut-out method when no matting model is installed", async () => {
    const { comfy, workDir } = await setup({ render: () => encodePng(discOnGreen(768, 768)) });
    const { llm } = stubLlm({
      plan: { ...PLAN, checklist: [{ id: "subject", text: "One red disc" }], key_color: "#00ff00" },
      critic: () => ({ items: [{ id: "subject", pass: true }] }),
    });
    const res = await generateAsset(
      { prompt: "a red disc icon", width: 256, height: 256, transparent: true },
      { comfy, llm, plannerModel: "test/planner", criticModel: "test/critic", workDir },
    );
    expect(res.trace.server.cutout).toEqual({ method: "chroma-key" });
    expect(res.trace.rounds[0]!.cutout).toMatchObject({ method: "chroma-key", backdrop: { requested: "#00FF00" } });
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

  describe("with split-loader model sets installed", () => {
    const familyInfo = () => ({ ...makeObjectInfo(), ...structuredClone(FAMILY_NODES) });

    it("defaults to the strongest installed family with its fixed distilled settings", async () => {
      const { fake, comfy, workDir } = await setup({ objectInfo: familyInfo(), render: () => lanternPng(768, 768) });
      const { llm } = stubLlm({ plan: PLAN, critic: () => ({ items: [{ id: "count", pass: true }, { id: "background", pass: true }] }) });
      const res = await generateAsset({ prompt: "lantern", width: 768, height: 768, seed: 5 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic" });
      const g = fake.prompts[0]!.graph;
      expect(g.unet!.inputs.unet_name).toBe("flux-2-klein-4b.safetensors");
      // The plan asked for 20 steps at cfg 6; the distilled model keeps 4 steps at cfg 1.
      expect(g.sigmas!.inputs).toMatchObject({ steps: 4, width: 768, height: 768 });
      expect(g.guider!.inputs.cfg).toBe(1);
      expect(g.noise!.inputs.noise_seed).toBe(5);
      expect(res.trace.server.family).toBe("flux2_klein");
    });

    it("uses a named family with sentence prompts and the key colour as a sentence", async () => {
      const { fake, comfy, workDir } = await setup({ objectInfo: familyInfo(), render: () => encodePng(discOnGreen(768, 768)) });
      const { llm, log } = stubLlm({ plan: { ...PLAN, key_color: "#00FF00" }, critic: () => ({ items: [{ id: "count", pass: true }, { id: "background", pass: true }] }) });
      const res = await generateAsset(
        { prompt: "lantern", width: 512, height: 512, transparent: true, seed: 5, checkpoint: "z_image_turbo_int8_convrot.safetensors" },
        { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic" },
      );
      const g = fake.prompts[0]!.graph;
      expect(g.unet!.inputs.unet_name).toBe("z_image_turbo_int8_convrot.safetensors");
      // The plan asked for 20 steps at cfg 6; the distilled model keeps 8 steps at cfg 1.
      expect(g.sampler!.inputs).toMatchObject({ steps: 8, cfg: 1, sampler_name: "res_multistep", seed: 5 });
      expect(String(g.pos!.inputs.text)).toMatch(/^red paper lantern, white background\. The subject is isolated on a plain, flat, uniform bright chroma-key green background/);
      expect(res.trace.server).toMatchObject({ family: "zimage", editModel: { family: "flux2_klein", model: "flux-2-klein-4b.safetensors" } });
      expect(positiveOf(log[0]!.body)).toContain("Model family: zimage");
      expect(positiveOf(log[0]!.body)).toContain("Instruction editing is available");
    });

    it("repairs with edit_previous, refuses tuning a distilled model, and re-prompts from text-to-image", async () => {
      const { fake, comfy, workDir } = await setup({ objectInfo: familyInfo(), render: () => lanternPng(768, 768) });
      const { llm, log } = stubLlm({
        plan: PLAN,
        critic: (n) => ({ items: [{ id: "count", pass: n >= 2 }, { id: "background", pass: true }] }),
        repair: (n) =>
          n === 0
            ? { rationale: "remove extras", edits: [{ op: "edit_previous", instruction: "Remove the extra lanterns" }, { op: "set_cfg", cfg: 7 }] }
            : { rationale: "start over", edits: [{ op: "set_prompt", text: "exactly one red paper lantern" }] },
      });
      // Z-Image generates; FLUX.2 klein, the installed edit model, edits.
      const res = await generateAsset(
        { prompt: "lantern", width: 768, height: 768, seed: 9, checkpoint: "z_image_turbo_int8_convrot.safetensors" },
        { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic", limits: { maxRounds: 3 } },
      );
      expect(res.trace.stopReason).toBe("all-pass");
      expect(fake.uploads).toHaveLength(1);
      const edit = fake.prompts[1]!.graph;
      expect(edit.load!.inputs.image).toBe(fake.uploads[0]!.name);
      expect(edit.refPos!.class_type).toBe("ReferenceLatent");
      expect(edit.pos!.inputs.text).toBe("Remove the extra lanterns");
      expect(edit.noise!.inputs.noise_seed).toBe(9);
      expect(res.trace.rounds[0]!.repair!.rejected[0]!.error).toMatch(/distilled/);
      expect(res.trace.rounds[1]!.settings).toMatchObject({ kind: "edit", family: "flux2_klein" });
      // After an edit, set_prompt goes back to text-to-image with the generating model.
      const third = fake.prompts[2]!.graph;
      expect(third.unet!.inputs.unet_name).toBe("z_image_turbo_int8_convrot.safetensors");
      expect(third.pos!.inputs.text).toBe("exactly one red paper lantern");
      expect(positiveOf(log.find((l) => l.tool === REPAIR_TOOL)!.body)).toContain("Instruction editing: available (flux2_klein");
    });

    it("tells the planner when edit_previous is not possible", async () => {
      const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768) });
      const { llm } = stubLlm({
        plan: PLAN,
        critic: (n) => ({ items: [{ id: "count", pass: n > 0 }, { id: "background", pass: true }] }),
        repair: () => ({ rationale: "edit", edits: [{ op: "edit_previous", instruction: "Remove the extra lanterns" }] }),
      });
      const res = await generateAsset({ prompt: "lantern", width: 768, height: 768 }, { comfy, llm, workDir, plannerModel: "test/planner", criticModel: "test/critic", limits: { maxRounds: 2 } });
      expect(res.trace.rounds[0]!.repair!.rejected[0]!.error).toMatch(/no instruction-editing model is installed/);
    });
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

  it("does not spend the time budget on waiting in a shared queue", async () => {
    // Every prompt waits 600 ms behind other clients, then runs in 5 ms. Counting
    // the waits, the second round would already exceed maxTimeMs.
    const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768), queueDelayMs: 600, runDelayMs: 5 });
    const { llm } = stubLlm({ plan: PLAN, critic: () => ({ items: [{ id: "count", pass: false }, { id: "background", pass: true }] }) });
    const res = await generateAsset(
      { prompt: "lantern", width: 768, height: 768 },
      { comfy, llm, workDir, limits: { maxRounds: 3, maxTimeMs: 900 }, plannerModel: "test/planner", criticModel: "test/critic" },
    );
    expect(res.trace.stopReason).toBe("max-rounds");
    expect(res.trace.rounds).toHaveLength(3);
    for (const r of res.trace.rounds) {
      expect(r.timing.queueWaitMs).toBeGreaterThanOrEqual(500);
      expect(r.timing.comfyMs).toBeGreaterThanOrEqual(r.timing.queueWaitMs);
    }
    expect(res.trace.totals.queueWaitMs).toBeGreaterThanOrEqual(1500);
    expect(res.trace.limits.maxWallTimeMs).toBe(60 * 60 * 1000);
  });

  it("stops with queue-timeout when the wall-clock ceiling runs out in the queue", async () => {
    const { comfy, workDir } = await setup({ render: () => lanternPng(768, 768), queueDelayMs: 600, runDelayMs: 5 });
    const { llm } = stubLlm({ plan: PLAN, critic: () => ({ items: [{ id: "count", pass: false }, { id: "background", pass: true }] }) });
    const res = await generateAsset(
      { prompt: "lantern", width: 768, height: 768 },
      { comfy, llm, workDir, limits: { maxRounds: 3, maxTimeMs: 60_000, maxWallTimeMs: 900 }, plannerModel: "test/planner", criticModel: "test/critic" },
    );
    expect(res.trace.stopReason).toBe("queue-timeout");
    expect(res.trace.best?.round).toBe(1);
    expect(res.trace.rounds[1]!.error?.message).toMatch(/still waiting in the queue/);
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
