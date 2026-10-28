// Unit tests for the design-asset eval: scoring, statistics, resumability,
// aggregation and report rendering. Stub data only; no live services.

import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { renderContactSheet, renderJudgeSheet } from "../evals/design-assets/html.js";
import { alignJudgement, JUDGE_TOOL, judgeImage, judgeUserText, CUTOUT_EXPLANATION, modelFamily, selectJudgeModel } from "../evals/design-assets/judge.js";
import { writeReports } from "../evals/design-assets/report.js";
import {
  ARMS,
  DEFAULT_ARMS,
  appendResult,
  armApplies,
  judgesToRedo,
  parseResults,
  pendingJobs,
  readResults,
  scoredCheckIds,
  type Arm,
  type JudgeResult,
  type ResultRow,
} from "../evals/design-assets/results.js";
import { bootstrapMeanCi, cohensKappa, comparePaired, median, quantile, signTest } from "../evals/design-assets/stats.js";
import { armStats, judgeAgreement, modelsReport, pairedReport, renderSummaryMarkdown, scoreRow, summarize } from "../evals/design-assets/summary.js";
import { checkpointOf, executionMs, foreignCheckpoints, llmModelsOf } from "../evals/design-assets/timing.js";
import { DEFAULT_RELIABILITY_SUBSET, loadTasks, parseTasks, selectTasks, type EvalTask } from "../evals/design-assets/tasks.js";
import { runArm } from "../evals/design-assets/arms.js";
import { judgeSet, readSetResults, appendSetResult, setMembers, setReferencesFor, SET_JUDGE_TOOL } from "../evals/design-assets/sets.js";
import { createImage, encodePng } from "../src/image.js";
import { MATTING_NODES } from "./fixtures/object-info-matting.js";
import { ComfyClient } from "../src/client.js";
import { AiandClient } from "../src/llm.js";
import { ModelSelectionError } from "../src/errors.js";
import type { AiandModel } from "../src/models.js";
import { solidPng, startFakeComfy } from "./fixtures/fake-comfy.js";
import { makeObjectInfo } from "./fixtures/object-info.js";
import { FAMILY_NODES } from "./fixtures/object-info-families.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  await Promise.all(cleanups.splice(0).map((c) => c()));
});

async function tempDir(): Promise<string> {
  const dir = await mkdtemp(path.join(os.tmpdir(), "comfy-agent-eval-"));
  cleanups.push(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

const TASK: EvalTask = {
  id: "t1",
  group: "icon",
  width: 512,
  height: 512,
  transparent: true,
  prompt: "a lantern",
  checklist: ["One lantern", "Red", "No text", "Centred", "Clean edges"],
};

function judge(passes: boolean[], model = "j/one", extra: Partial<JudgeResult> = {}): JudgeResult {
  return {
    model,
    shape: "openai",
    items: passes.map((pass, i) => ({ id: `c${i + 1}`, text: `item ${i + 1}`, pass })),
    retries: 0,
    inputTokens: 100,
    outputTokens: 20,
    costUsd: 0.001,
    latencyMs: 5,
    ...extra,
  };
}

function row(p: { taskId?: string; group?: string; arm: Arm; seed?: number; passes?: boolean[]; checksPass?: boolean; transparent?: boolean } & Partial<ResultRow>): ResultRow {
  const transparent = p.transparent ?? false;
  const checkPass = p.checksPass ?? true;
  const { passes, checksPass: _c, ...rest } = p;
  return {
    taskId: "t1",
    group: "icon",
    seed: 1,
    width: 512,
    height: 512,
    status: "ok",
    rounds: p.arm === "loop" ? 2 : 1,
    wallMs: 10_000,
    comfyMs: 8_000,
    llm: { inputTokens: 0, outputTokens: 0, costUsd: p.arm === "raw" ? 0 : 0.02, estimatedCostUsd: 0, reportedCostUsd: null },
    checks: scoredCheckIds(transparent).map((id) => ({ id, pass: checkPass, detail: "" })),
    scoredChecks: scoredCheckIds(transparent),
    finishedAt: "2026-10-28T00:00:00.000Z",
    ...(passes ? { judge: judge(passes) } : {}),
    ...rest,
    transparent,
  };
}

const ALL = [true, true, true, true, true];

// ---------------------------------------------------------------------------

describe("task set", () => {
  it("loads the committed tasks: 44 tasks, 5 groups, 5 checklist items each; 20 icons including two sets of 4", async () => {
    const tasks = await loadTasks();
    expect(tasks).toHaveLength(44);
    expect(new Set(tasks.map((t) => t.group)).size).toBe(5);
    expect(tasks.every((t) => t.checklist.length === 5)).toBe(true);
    expect(tasks.filter((t) => t.group === "icon")).toHaveLength(20);
    expect(setMembers(tasks, "glossy-ui").map((t) => t.id)).toEqual(["set-glossy-chat", "set-glossy-calendar", "set-glossy-bell", "set-glossy-lock"]);
    expect(setMembers(tasks, "cafe")).toHaveLength(4);
    expect(tasks.filter((t) => t.set).every((t) => t.group === "icon" && t.transparent)).toBe(true);
  });

  it("runs each ablation only on the tasks it measures, and scores the right cut-out check", () => {
    const icon = { transparent: true, group: "icon" };
    const setIcon = { ...icon, set: "cafe" };
    const hero = { transparent: false, group: "hero" };
    expect(ARMS.filter((a) => armApplies(a, hero))).toEqual(["raw", "plan", "loop"]);
    expect(ARMS.filter((a) => armApplies(a, icon))).toEqual(["raw", "plan", "loop", "loop-chroma", "loop-noref"]);
    expect(ARMS.filter((a) => armApplies(a, setIcon))).toEqual([...ARMS]);
    expect(DEFAULT_ARMS).toEqual(["raw", "plan", "loop"]);
    expect(scoredCheckIds(false)).toEqual(["size"]);
    expect(scoredCheckIds(true)).toEqual(["size", "alpha", "key_residue"]);
    expect(scoredCheckIds(true, "matte", true)).toEqual(["size", "alpha", "matte_residue", "reference_copy"]);
  });

  it("has a reliability subset of 8 known tasks that covers every group and some transparent tasks", async () => {
    const tasks = await loadTasks();
    const subset = tasks.filter((t) => (DEFAULT_RELIABILITY_SUBSET as readonly string[]).includes(t.id));
    expect(subset).toHaveLength(8);
    expect(new Set(subset.map((t) => t.group))).toEqual(new Set(tasks.map((t) => t.group)));
    expect(subset.some((t) => t.transparent)).toBe(true);
  });

  it("rejects malformed tasks and unknown selections", () => {
    expect(() => parseTasks({ tasks: [{ id: "x", group: "g", width: 10, height: 64, prompt: "p", checklist: ["a"] }] })).toThrow(/width/);
    expect(() => parseTasks({ tasks: [{ ...TASK }, { ...TASK }] })).toThrow(/duplicate/);
    expect(() => selectTasks([TASK], { ids: ["nope"] })).toThrow(/unknown task/);
    expect(selectTasks([TASK, { ...TASK, id: "t2", group: "hero" }], { groups: ["hero"] }).map((t) => t.id)).toEqual(["t2"]);
  });
});

describe("statistics", () => {
  it("computes Cohen's kappa against a worked example", () => {
    // 50 items: both pass 20, A pass/B fail 5, A fail/B pass 10, both fail 15.
    const a = [...Array(20).fill(true), ...Array(5).fill(true), ...Array(10).fill(false), ...Array(15).fill(false)];
    const b = [...Array(20).fill(true), ...Array(5).fill(false), ...Array(10).fill(true), ...Array(15).fill(false)];
    const k = cohensKappa(a, b);
    expect(k.po).toBeCloseTo(0.7, 10);
    expect(k.pe).toBeCloseTo(0.5, 10);
    expect(k.kappa).toBeCloseTo(0.4, 10);
  });

  it("gives kappa 1 for perfect agreement and leaves it undefined when both raters are constant", () => {
    expect(cohensKappa([true, false, true], [true, false, true]).kappa).toBe(1);
    expect(cohensKappa([true, true], [true, true]).kappa).toBeNull();
    expect(cohensKappa([], []).kappa).toBeNull();
    expect(() => cohensKappa([true], [])).toThrow();
  });

  it("computes the exact two-sided sign test", () => {
    expect(signTest(8, 2)).toBeCloseTo((2 * (1 + 10 + 45)) / 1024, 12);
    expect(signTest(5, 5)).toBe(1);
    expect(signTest(0, 0)).toBe(1);
    expect(signTest(10, 0)).toBeCloseTo(2 / 1024, 12);
    expect(signTest(3, 7)).toBe(signTest(7, 3));
  });

  it("bootstraps a reproducible CI that brackets the mean", () => {
    const diffs = [0.2, 0, 0.4, -0.2, 0.2, 0.6, 0, 0.2];
    const a = bootstrapMeanCi(diffs, { seed: 7 });
    const b = bootstrapMeanCi(diffs, { seed: 7 });
    expect(a).toEqual(b);
    expect(a!.low).toBeLessThan(0.175);
    expect(a!.high).toBeGreaterThan(0.175);
    const flat = bootstrapMeanCi([0.2, 0.2, 0.2])!;
    expect(flat.low).toBeCloseTo(0.2, 12);
    expect(flat.high).toBeCloseTo(0.2, 12);
    expect(bootstrapMeanCi([])).toBeNull();
  });

  it("counts improved, tied and regressed pairs", () => {
    const c = comparePaired([{ a: 0.2, b: 0.6 }, { a: 0.4, b: 0.4 }, { a: 0.8, b: 0.6 }, { a: 0, b: 1 }]);
    expect([c.improved, c.tied, c.regressed, c.n]).toEqual([2, 1, 1, 4]);
    expect(c.meanDiff).toBeCloseTo((0.4 + 0 - 0.2 + 1) / 4, 12);
    expect(c.signTestP).toBeCloseTo(1, 12);
  });

  it("computes medians and quantiles", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 2, 3])).toBe(2.5);
    expect(median([])).toBeNull();
    expect(quantile([0, 10], 0.25)).toBe(2.5);
  });
});

describe("results file and resumability", () => {
  it("keeps the last row per key and skips lines cut short by a crash", () => {
    const first = row({ arm: "raw", status: "error", error: { class: "X", message: "boom", stage: "generate" } });
    const second = row({ arm: "raw", passes: ALL });
    const rows = parseResults(`${JSON.stringify(first)}\n${JSON.stringify(row({ arm: "loop", passes: ALL }))}\n${JSON.stringify(second)}\n{"taskId":"t1","ar`);
    expect(rows).toHaveLength(2);
    expect(rows.find((r) => r.arm === "raw")!.status).toBe("ok");
  });

  it("skips finished task × arm × seed rows and reruns failed ones only on request", async () => {
    const dir = await tempDir();
    const file = path.join(dir, "results.jsonl");
    expect(await readResults(file)).toEqual([]);
    await appendResult(file, row({ arm: "raw", passes: ALL }));
    await appendResult(file, row({ arm: "plan", status: "error", error: { class: "ComfyTimeoutError", message: "t", stage: "generate" } }));
    const existing = await readResults(file);
    const jobs = (["raw", "plan", "loop"] as const).flatMap((arm) => [1, 2].map((seed) => ({ taskId: "t1", arm, seed })));
    expect(pendingJobs(jobs, existing).map((j) => `${j.arm}${j.seed}`)).toEqual(["raw2", "plan2", "loop1", "loop2"]);
    expect(pendingJobs(jobs, existing, { retryFailed: true }).map((j) => `${j.arm}${j.seed}`)).toEqual(["raw2", "plan1", "plan2", "loop1", "loop2"]);
  });

  it("reruns rows that stopped for a listed reason", () => {
    const existing = [row({ arm: "loop", passes: ALL, stopReason: "time-limit" }), row({ arm: "plan", passes: ALL, stopReason: "max-rounds" })];
    const jobs = (["plan", "loop"] as const).map((arm) => ({ taskId: "t1", arm, seed: 1 }));
    expect(pendingJobs(jobs, existing)).toEqual([]);
    expect(pendingJobs(jobs, existing, { rerunStopReasons: ["time-limit"] }).map((j) => j.arm)).toEqual(["loop"]);
  });

  it("redoes only missing or failed grades on generated rows", () => {
    const ok = row({ arm: "raw", passes: ALL, judgeViewPath: "/x.png" });
    expect(judgesToRedo(ok, false, true)).toEqual([]);
    expect(judgesToRedo(ok, true, true)).toEqual(["judge2"]);
    expect(judgesToRedo(ok, true, false)).toEqual([]);
    expect(judgesToRedo({ ...ok, judge: judge([], "j", { error: "x" }) }, false, true)).toEqual(["judge"]);
    expect(judgesToRedo({ ...ok, status: "error" }, true, true)).toEqual([]);
  });
});

describe("judge", () => {
  it("grades the fixed checklist by stable ids and explains cut-outs", () => {
    const text = judgeUserText(TASK);
    expect(text).toContain("- c1: One lantern");
    expect(text).toContain("- c5: Clean edges");
    expect(text).toContain(CUTOUT_EXPLANATION);
    expect(judgeUserText({ ...TASK, transparent: false })).not.toContain("magenta divider");
  });

  it("aligns tolerant model output to the checklist and fails what is missing", () => {
    const a = alignJudgement(TASK, { items: JSON.stringify([{ id: "c1", pass: "yes" }, { id: "c3", pass: false, evidence: "has text" }]) });
    expect(a.items.map((i) => i.pass)).toEqual([true, false, false, false, false]);
    expect(a.ungraded).toEqual(["c2", "c4", "c5"]);
    expect(a.items[2]!.evidence).toBe("has text");
    const positional = alignJudgement(TASK, { items: ALL.map((p, i) => ({ id: `item${i}`, pass: p })) });
    expect(positional.ungraded).toEqual([]);
  });

  function stubJudgeLlm(responses: (() => Response)[]) {
    const calls: Record<string, unknown>[] = [];
    const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
      if (String(input).endsWith("/models")) return new Response(JSON.stringify({ data: [{ id: "j/one", input_per_1m: 1, output_per_1m: 1 }] }));
      calls.push(JSON.parse(String(init!.body)));
      const next = responses.shift();
      if (!next) throw new Error("unexpected call");
      return next();
    }) as typeof fetch;
    return { llm: new AiandClient({ apiKey: "test-key", baseUrl: "https://llm.test/v1", fetch: fetchImpl, maxRetries: 0 }), calls };
  }
  const toolReply = (args: unknown) => () =>
    new Response(JSON.stringify({ choices: [{ message: { tool_calls: [{ type: "function", function: { name: JUDGE_TOOL, arguments: JSON.stringify(args) } }] } }], usage: { prompt_tokens: 1000, completion_tokens: 100 } }));

  it("asks once more naming every id when grades are missing, and keeps the better answer", async () => {
    const { llm, calls } = stubJudgeLlm([
      toolReply({ id: "c1", pass: true }),
      toolReply({ items: ALL.map((p, i) => ({ id: `c${i + 1}`, pass: p, evidence: "ok" })) }),
    ]);
    const res = await judgeImage(TASK, Buffer.from("png"), { llm, model: "j/one", shape: "openai" });
    expect(calls).toHaveLength(2);
    expect(JSON.stringify(calls[1])).toContain("exactly one entry for each of these ids: c1, c2, c3, c4, c5");
    expect(res.error).toBeUndefined();
    expect(res.items.every((i) => i.pass)).toBe(true);
    expect(res.inputTokens).toBe(2000);
    expect(res.costUsd).toBeCloseTo(0.0022, 10);
  });

  it("retries transport errors and reports a judge error instead of throwing", async () => {
    const fail = () => new Response("overloaded", { status: 503 });
    const { llm, calls } = stubJudgeLlm([fail, fail, fail]);
    const res = await judgeImage(TASK, Buffer.from("png"), { llm, model: "j/one", shape: "openai", retryDelayMs: 1 });
    expect(calls).toHaveLength(3);
    expect(res.error).toMatch(/503/);
    expect(res.items).toEqual([]);
  });
});

describe("scoring and aggregation", () => {
  it("scores full pass as every item plus the scored code checks", () => {
    expect(scoreRow(row({ arm: "raw", passes: ALL })).fullPass).toBe(true);
    expect(scoreRow(row({ arm: "raw", passes: [true, true, true, true, false] }))).toMatchObject({ fullPass: false, itemPassRate: 0.8 });
    expect(scoreRow(row({ arm: "raw", passes: ALL, checksPass: false, transparent: true }))).toMatchObject({ fullPass: false, codePass: false, itemPassRate: 1 });
    // A generation failure scores zero; a judge failure is excluded.
    expect(scoreRow(row({ arm: "raw", status: "error" }))).toMatchObject({ scorable: true, itemPassRate: 0, fullPass: false });
    expect(scoreRow(row({ arm: "raw", judge: judge([], "j", { error: "x" }) })).scorable).toBe(false);
  });

  it("ignores checks that are not scored (flat background)", () => {
    const r = row({ arm: "raw", passes: ALL, transparent: true });
    r.checks.push({ id: "flat_background", pass: false, detail: "" });
    expect(scoreRow(r).fullPass).toBe(true);
  });

  const fixture = (): ResultRow[] => [
    row({ taskId: "a", group: "hero", arm: "raw", passes: [true, false, false, true, true] }),
    row({ taskId: "a", group: "hero", arm: "plan", passes: [true, true, false, true, true] }),
    row({ taskId: "a", group: "hero", arm: "loop", passes: ALL, wallMs: 60_000, rounds: 3 }),
    row({ taskId: "b", group: "icon", arm: "raw", passes: ALL }),
    row({ taskId: "b", group: "icon", arm: "plan", passes: ALL }),
    row({ taskId: "b", group: "icon", arm: "loop", passes: [true, true, true, true, false], wallMs: 30_000 }),
    row({ taskId: "c", group: "icon", arm: "raw", passes: [false, false, false, false, false] }),
    row({ taskId: "c", group: "icon", arm: "plan", status: "error", error: { class: "LlmHttpError", message: "500", stage: "generate" } }),
    row({ taskId: "c", group: "icon", arm: "loop", passes: [true, true, true, false, false], judge2: judge([true, true, false, false, false], "j/two") }),
  ];

  it("builds per-arm statistics", () => {
    const s = armStats("loop", fixture());
    expect(s.rows).toBe(3);
    expect(s.fullPassRate).toBeCloseTo(1 / 3, 12);
    expect(s.meanItemPassRate).toBeCloseTo((1 + 0.8 + 0.6) / 3, 12);
    expect(s.wallSeconds.median).toBe(30);
    expect(s.rounds.mean).toBeCloseTo((3 + 2 + 2) / 3, 12);
    const plan = armStats("plan", fixture());
    expect(plan.generationErrors).toBe(1);
    expect(plan.meanItemPassRate).toBeCloseTo((0.8 + 1 + 0) / 3, 12);
    expect(armStats("raw", fixture()).totalCostUsd).toBe(0);
  });

  it("pairs tasks across arms", () => {
    const p = pairedReport(fixture(), "raw", "loop");
    expect(p.itemPassRate).toMatchObject({ n: 3, improved: 2, tied: 0, regressed: 1 });
    expect(p.fullPass).toMatchObject({ improved: 1, tied: 1, regressed: 1 });
  });

  it("measures agreement between the two judges on the rows both graded", () => {
    const j = judgeAgreement(fixture());
    expect(j.images).toBe(1);
    expect(j.items.n).toBe(5);
    expect(j.items.agree).toBe(4);
    expect(j.judge2).toBe("j/two");
    expect(j.perItem.find((p) => p.itemId === "c3")).toMatchObject({ n: 1, agree: 0 });
  });

  it("renders the summary tables per arm, per group, paired and agreement", () => {
    const s = summarize(fixture(), new Date("2026-10-28T00:00:00Z"));
    expect(s.perGroup.map((g) => g.group)).toEqual(["hero", "icon"]);
    expect(s.paired.map((p) => `${p.b}-${p.a}`)).toEqual(["loop-raw", "loop-plan", "plan-raw"]);
    expect(s.errors).toHaveLength(1);
    const md = renderSummaryMarkdown(s);
    for (const h of ["## Per arm", "## Per group", "### hero", "## Paired comparisons", "## Judge agreement", "## Errors"]) expect(md).toContain(h);
    expect(md).toContain("| C loop vs A raw | item pass rate | 3 | 2 | 0 | 1 |");
    expect(md).toContain("| A raw | 3 | 3 | 0 | 0 | 33.3% |");
  });
});

describe("judge model selection", () => {
  const m = (id: string, capabilities = ["vision", "tool_calling"], extra: Partial<AiandModel> = {}): AiandModel => ({ id, capabilities, reasoningEfforts: [], ...extra });
  const catalog = [
    m("moonshotai/kimi-k3"),
    m("zai-org/glm-5.3-flash"),
    m("zai-org/glm-5.3", ["reasoning", "tool_calling"]),
    m("qwen/qwen3.8-27b"),
    m("google/gemma-4-31b-it"),
    m("google/gemma-old", ["vision", "tool_calling"], { deprecationDate: "2020-01-01" }),
  ];
  const exclude = ["moonshotai", "zai-org"];

  it("takes the preferred judge when the catalog offers it from an independent family", () => {
    expect(selectJudgeModel(catalog, { preferred: ["google/gemma-4-31b-it"], excludeFamilies: exclude })).toBe("google/gemma-4-31b-it");
  });

  it("never picks a model from the planner's or critic's family, even as a fallback", () => {
    const picked = selectJudgeModel(catalog, { preferred: ["zai-org/glm-5.3-flash", "nobody/missing"], excludeFamilies: exclude });
    expect(["qwen/qwen3.8-27b", "google/gemma-4-31b-it"]).toContain(picked);
  });

  it("picks a second judge outside the first judge's family too", () => {
    expect(selectJudgeModel(catalog, { preferred: ["google/gemma-4-31b-it"], excludeFamilies: [...exclude, "google"] })).toBe("qwen/qwen3.8-27b");
  });

  it("refuses a configured judge from an excluded family, and validates others against the catalog", () => {
    expect(() => selectJudgeModel(catalog, { configured: "zai-org/glm-5.3-flash", preferred: [], excludeFamilies: exclude })).toThrow(/family/);
    expect(() => selectJudgeModel(catalog, { configured: "google/gemma-old", preferred: [], excludeFamilies: exclude })).toThrow(ModelSelectionError);
    expect(() => selectJudgeModel(catalog, { preferred: [], excludeFamilies: ["moonshotai", "zai-org", "qwen", "google"] })).toThrow(ModelSelectionError);
  });

  it("reads the family from the provider field, else the id namespace", () => {
    expect(modelFamily("google/gemma-4-31b-it")).toBe("google");
    expect(modelFamily("gemma-local", "Google")).toBe("google");
    expect(modelFamily("plain")).toBe("plain");
  });
});

describe("model consistency across a resumed run", () => {
  it("lists image models in the run that differ from the current default", () => {
    expect(foreignCheckpoints(["sd_xl_base_1.0.safetensors", undefined, "sd_xl_base_1.0.safetensors"], "sd_xl_base_1.0.safetensors")).toEqual([]);
    expect(foreignCheckpoints(["sd_xl_base_1.0.safetensors", "flux2-klein.safetensors"], "flux2-klein.safetensors")).toEqual(["sd_xl_base_1.0.safetensors"]);
  });

  it("recovers the checkpoint of an older row from its raw or loop trace", async () => {
    const dir = await tempDir();
    const rawTrace = path.join(dir, "raw.json");
    const loopTrace = path.join(dir, "loop.json");
    await writeFile(rawTrace, JSON.stringify({ checkpoint: "a.safetensors" }));
    await writeFile(loopTrace, JSON.stringify({ server: { checkpoint: "b.safetensors" }, models: { planner: "p/one", critic: "c/one" } }));
    expect(await llmModelsOf(row({ arm: "loop", tracePath: loopTrace }))).toEqual({ planner: "p/one", critic: "c/one" });
    expect(await llmModelsOf(row({ arm: "raw", tracePath: rawTrace }))).toBeUndefined();
    expect(await checkpointOf(row({ arm: "raw", tracePath: rawTrace }))).toBe("a.safetensors");
    expect(await checkpointOf(row({ arm: "loop", tracePath: loopTrace }))).toBe("b.safetensors");
    expect(await checkpointOf(row({ arm: "loop", checkpoint: "c.safetensors", tracePath: loopTrace }))).toBe("c.safetensors");
    expect(await checkpointOf(row({ arm: "loop", tracePath: path.join(dir, "missing.json") }))).toBeUndefined();
  });

  it("records the image model and the LLM models in the summary", () => {
    const rows = [
      row({ arm: "raw", passes: ALL, checkpoint: "sd_xl_base_1.0.safetensors" }),
      row({ arm: "loop", passes: ALL, checkpoint: "sd_xl_base_1.0.safetensors", models: { planner: "p/one", critic: "c/one" }, judge2: judge(ALL, "j/two") }),
    ];
    const models = modelsReport(rows);
    expect(models.imageModel).toEqual([{ id: "sd_xl_base_1.0.safetensors", rows: 2, arms: ["raw", "loop"] }]);
    expect(models.planner).toEqual([{ id: "p/one", rows: 1 }]);
    expect(models.judge).toEqual([{ id: "j/one", rows: 2 }]);
    expect(models.judge2).toEqual([{ id: "j/two", rows: 1 }]);
    const s = summarize(rows);
    expect(s.models.imageModel[0]!.id).toBe("sd_xl_base_1.0.safetensors");
    expect(renderSummaryMarkdown(s)).toContain("Image model (ComfyUI checkpoint): sd_xl_base_1.0.safetensors. Planner: p/one. Critic: c/one. Judge: j/one.");
  });
});

describe("ComfyUI timing", () => {
  it("reads execution time from history timestamps, excluding queue wait", () => {
    const entry = (messages: [string, Record<string, unknown>][]) => ({ outputs: {}, status: { messages } });
    expect(executionMs(entry([["execution_start", { timestamp: 1000 }], ["execution_cached", { timestamp: 1001 }], ["execution_success", { timestamp: 31_000 }]]))).toBe(30_000);
    expect(executionMs(entry([["execution_start", { timestamp: 1000 }], ["execution_error", { timestamp: 4000 }]]))).toBe(3000);
    expect(executionMs(entry([["execution_start", { timestamp: 1000 }]]))).toBeUndefined();
    expect(executionMs(null)).toBeUndefined();
  });

  it("reports wall time without queue wait and stop reasons per arm", () => {
    const rows = [
      row({ arm: "loop", passes: ALL, wallMs: 100_000, comfyMs: 90_000, comfyExecMs: 30_000, comfyQueueMs: 60_000, stopReason: "time-limit" }),
      row({ taskId: "t2", arm: "loop", passes: ALL, wallMs: 50_000, comfyMs: 40_000, comfyExecMs: 40_000, comfyQueueMs: 0, stopReason: "all-pass" }),
    ];
    const s = armStats("loop", rows);
    expect(s.wallExclQueueSeconds.mean).toBe(45);
    expect(s.comfyExecSeconds.mean).toBe(35);
    expect(s.queueWaitSecondsTotal).toBe(60);
    expect(s.stopReasons).toEqual({ "time-limit": 1, "all-pass": 1 });
    expect(renderSummaryMarkdown(summarize(rows))).toContain("Time queued behind other ComfyUI clients: C loop 60 s.");
  });
});

describe("reports", () => {
  it("writes self-contained HTML with escaped text and no external assets", async () => {
    const dir = await tempDir();
    const rows = [row({ taskId: "t1", arm: "raw", passes: ALL, judge2: judge(ALL, "j/two"), imagePath: path.join(dir, "missing.png") })];
    const task = { ...TASK, prompt: 'lantern <script>alert("x")</script>' };
    const html = renderContactSheet([task], rows, new Map([["t1|raw|1", "data:image/png;base64,AAAA"]]), dir);
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toMatch(/(src|href)="https?:/);
    expect(html).toContain('href="missing.png"');
    const sheet = renderJudgeSheet([task], rows, new Map(), dir);
    expect(sheet).toContain("j/two");
    expect(sheet).not.toMatch(/(src|href)="https?:/);
  });

  it("writes every report file from rows", async () => {
    const dir = await tempDir();
    const files = await writeReports(dir, [TASK], [row({ arm: "raw", passes: ALL }), row({ arm: "loop", passes: ALL })]);
    expect(files).toEqual(["summary.md", "summary.json", "contact-sheet.html", "judge-subset.html"]);
    const json = JSON.parse(await readFile(path.join(dir, "summary.json"), "utf8"));
    expect(json.perArm.map((a: { arm: string }) => a.arm)).toEqual(["raw", "loop"]);
  });
});

describe("raw arm", () => {
  it("uses the server's default model, the same one the plan and loop arms use", async () => {
    const fake = await startFakeComfy({ objectInfo: { ...makeObjectInfo(), ...structuredClone(FAMILY_NODES) }, render: () => solidPng(512, 512) });
    const workDir = await mkdtemp(path.join(os.tmpdir(), "raw-arm-"));
    try {
      const task = (await loadTasks()).find((t) => !t.transparent)!;
      const out = await runArm("raw", task, 1, {
        comfy: new ComfyClient({ baseUrl: fake.url, WebSocket: null, pollIntervalMs: 10 }),
        llm: new AiandClient({ apiKey: "unused" }),
        plannerModel: "p",
        criticModel: "c",
        shape: "openai",
        loopMaxRounds: 1,
        timeoutMs: 10_000,
        workDir,
      });
      expect(out.row.status).toBe("ok");
      expect(out.row.checkpoint).toBe("flux-2-klein-4b.safetensors");
      const graph = fake.prompts[0]!.graph;
      expect(graph.unet!.inputs.unet_name).toBe("flux-2-klein-4b.safetensors");
      expect(Object.values(graph).some((n) => n.class_type === "CheckpointLoaderSimple")).toBe(false);
    } finally {
      await fake.close();
      await rm(workDir, { recursive: true, force: true });
    }
  });
});

describe("raw arm with a matting model", () => {
  it("cuts transparent tasks out with the matting model, like the loop, and scores matte_residue", async () => {
    const matted = createImage(512, 512, { r: 250, g: 250, b: 250, a: 0 });
    for (let y = 150; y < 350; y += 1) for (let x = 150; x < 350; x += 1) matted.data.set([200, 30, 40, 255], (y * 512 + x) * 4);
    const fake = await startFakeComfy({ objectInfo: { ...makeObjectInfo(), ...structuredClone(FAMILY_NODES), ...structuredClone(MATTING_NODES) }, render: () => encodePng(matted) });
    const workDir = await mkdtemp(path.join(os.tmpdir(), "raw-matte-"));
    try {
      const task = (await loadTasks()).find((t) => t.id === "icon-lantern")!;
      const out = await runArm("raw", task, 1, {
        comfy: new ComfyClient({ baseUrl: fake.url, WebSocket: null, pollIntervalMs: 10 }),
        llm: new AiandClient({ apiKey: "unused" }),
        plannerModel: "p",
        criticModel: "c",
        shape: "openai",
        loopMaxRounds: 1,
        timeoutMs: 10_000,
        workDir,
      });
      expect(out.row).toMatchObject({ status: "ok", cutout: "matte", scoredChecks: ["size", "alpha", "matte_residue"] });
      expect(out.row.checks.find((c) => c.id === "matte_residue")?.pass).toBe(true);
      expect(Object.values(fake.prompts[0]!.graph).some((n) => n.class_type === "RemoveBackground")).toBe(true);
      expect(String(fake.prompts[0]!.graph.pos!.inputs.text)).toContain("white background");
    } finally {
      await fake.close();
      await rm(workDir, { recursive: true, force: true });
    }
  });
});

describe("icon sets", () => {
  const tasks: EvalTask[] = [
    { id: "a", group: "icon", set: "s", width: 64, height: 64, transparent: true, prompt: "a", checklist: ["x"] },
    { id: "b", group: "icon", set: "s", width: 64, height: 64, transparent: true, prompt: "b", checklist: ["x"] },
    { id: "c", group: "icon", set: "s", width: 64, height: 64, transparent: true, prompt: "c", checklist: ["x"] },
  ];

  it("gives a member the earlier members of the same arm and seed as style references, newest first, and none for loop-noset", () => {
    const rows = [
      row({ taskId: "a", arm: "loop", imagePath: "/img/a-loop.png" }),
      row({ taskId: "b", arm: "loop", imagePath: "/img/b-loop.png" }),
      row({ taskId: "a", arm: "plan", imagePath: "/img/a-plan.png" }),
      row({ taskId: "a", arm: "loop", seed: 2, imagePath: "/img/a-loop-2.png" }),
    ];
    const read = (p: string) => Buffer.from(p);
    expect(setReferencesFor(tasks[2]!, "loop", 1, tasks, rows, read).map((r) => r.id)).toEqual(["b", "a"]);
    expect(setReferencesFor(tasks[1]!, "plan", 1, tasks, rows, read).map((r) => r.id)).toEqual(["a"]);
    expect(setReferencesFor(tasks[0]!, "loop", 1, tasks, rows, read)).toEqual([]);
    expect(setReferencesFor(tasks[2]!, "loop-noset", 1, tasks, rows, read)).toEqual([]);
  });

  it("judges a finished set from one sheet, names off-style members, and survives a broken judge", async () => {
    const calls: Record<string, unknown>[] = [];
    const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
      if (String(input).endsWith("/models")) return new Response(JSON.stringify({ data: [{ id: "judge/vision", capabilities: ["vision", "tool_calling"] }] }));
      const body = JSON.parse(String(init!.body)) as Record<string, unknown>;
      calls.push(body);
      const args = { consistent: false, score: 2, outliers: [3, 9], evidence: "the third icon is flat" };
      return new Response(JSON.stringify({ choices: [{ message: { tool_calls: [{ function: { name: SET_JUDGE_TOOL, arguments: JSON.stringify(args) } }] } }], usage: { prompt_tokens: 10, completion_tokens: 5 } }), {
        headers: { "X-Cost": "0.001", "X-Cost-Currency": "USD" },
      });
    }) as typeof fetch;
    const llm = new AiandClient({ apiKey: "k", baseUrl: "https://llm.test/v1", fetch: fetchImpl });
    const members = ["a", "b", "c"].map((id) => ({ id, image: createImage(32, 32, { r: 100, g: 100, b: 100 }) }));
    const j = await judgeSet("s", "loop", 1, members, { llm, model: "judge/vision", shape: "openai" });
    expect(j).toMatchObject({ set: "s", arm: "loop", seed: 1, members: ["a", "b", "c"], score: 2, consistent: false, outliers: ["c"], costUsd: 0.001 });
    expect(JSON.stringify(calls[0]!.messages).match(/data:image\/png;base64,/g)).toHaveLength(1);

    const broken = new AiandClient({ apiKey: "k", baseUrl: "https://llm.test/v1", fetch: (async () => new Response("no", { status: 500 })) as typeof fetch, maxRetries: 0 });
    expect((await judgeSet("s", "loop", 1, members, { llm: broken, model: "judge/vision", shape: "openai" })).error).toBeTruthy();
  });

  it("reports set consistency per arm in the summary, and pairs each ablation with the loop", async () => {
    const dir = await tempDir();
    const file = path.join(dir, "set-results.jsonl");
    const base = { members: ["a", "b"], model: "j", costUsd: 0, finishedAt: "2026-10-29T00:00:00Z" };
    await appendSetResult(file, { ...base, set: "s", arm: "loop", seed: 1, score: 5, consistent: true, outliers: [] });
    await appendSetResult(file, { ...base, set: "s", arm: "loop-noset", seed: 1, score: 2, consistent: false, outliers: ["b"] });
    const sets = await readSetResults(file);
    const md = renderSummaryMarkdown(summarize([row({ arm: "loop", passes: ALL }), row({ arm: "loop-noset", passes: ALL })], new Date(), sets));
    expect(md).toContain("## Icon set consistency");
    expect(md).toContain("| C loop | 1 | 100.0% | 5.00 |");
    expect(md).toContain("| C− no set refs | 1 | 0.0% | 2.00 |");
    expect(md).toContain("| C loop vs C− no set refs |");
  });
});
