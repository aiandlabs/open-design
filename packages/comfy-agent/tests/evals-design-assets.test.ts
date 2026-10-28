// Unit tests for the design-asset eval: scoring, statistics, resumability,
// aggregation and report rendering. Stub data only; no live services.

import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { renderContactSheet, renderJudgeSheet } from "../evals/design-assets/html.js";
import { alignJudgement, JUDGE_TOOL, judgeImage, judgeUserText, CUTOUT_EXPLANATION } from "../evals/design-assets/judge.js";
import { writeReports } from "../evals/design-assets/report.js";
import {
  appendResult,
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
import { armStats, judgeAgreement, pairedReport, renderSummaryMarkdown, scoreRow, summarize } from "../evals/design-assets/summary.js";
import { DEFAULT_RELIABILITY_SUBSET, loadTasks, parseTasks, selectTasks, type EvalTask } from "../evals/design-assets/tasks.js";
import { AiandClient } from "../src/llm.js";

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
  it("loads the committed tasks: 30 tasks, 5 groups, 5 checklist items each", async () => {
    const tasks = await loadTasks();
    expect(tasks).toHaveLength(30);
    expect(new Set(tasks.map((t) => t.group)).size).toBe(5);
    expect(tasks.every((t) => t.checklist.length === 5)).toBe(true);
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
    return { llm: new AiandClient({ apiKey: "test-key", baseUrl: "https://llm.test/v1", fetch: fetchImpl }), calls };
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
