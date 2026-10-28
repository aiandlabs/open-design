// The three arms. Each produces one final image for a task and seed:
//   raw   the user's prompt straight into the default text-to-image graph
//   plan  the planner plus one generation (maxRounds 1)
//   loop  the full asset loop
// Generation size, checkpoint, sampling defaults, post-processing and code
// checks are shared with the loop so the arms differ only in what they test.

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import type { ComfyClient } from "../../src/client.js";
import { buildTxt2Img, chooseCheckpoint, defaultSampling, detectFamily, planGenerationSize } from "../../src/graphs.js";
import { decodePng, encodePng, type RgbaImage } from "../../src/image.js";
import type { AiandClient, ApiShape } from "../../src/llm.js";
import { deterministicChecks, generateAsset, postProcess, type AssetRequest } from "../../src/loop.js";
import { composeKeyedPrompt, KEY_NEGATIVE } from "../../src/prompts.js";
import { scoredCheckIds, type Arm, type ResultRow } from "./results.js";
import type { EvalTask } from "./tasks.js";

/** Key colour arm A uses for transparent tasks (the loop's own fallback). */
export const RAW_KEY_COLOR = "#00FF00";

export type ArmContext = {
  comfy: ComfyClient;
  llm: AiandClient;
  /** Already resolved against the ai& catalog, so every arm and task uses the same ids. */
  plannerModel: string;
  criticModel: string;
  shape: ApiShape;
  plannerReasoningEffort?: string;
  criticReasoningEffort?: string;
  /** Rounds for the loop arm. */
  loopMaxRounds: number;
  /** Per task × arm deadline. */
  timeoutMs: number;
  /** Directory for this task × arm × seed. */
  workDir: string;
  onProgress?: (message: string) => void;
};

export type ArmOutput = {
  row: Omit<ResultRow, "judge" | "judge2" | "judgeViewPath" | "finishedAt">;
  image?: RgbaImage;
};

function request(task: EvalTask, seed: number): AssetRequest {
  return { prompt: task.prompt, width: task.width, height: task.height, transparent: task.transparent, seed };
}

function baseRow(task: EvalTask, arm: Arm, seed: number) {
  return {
    taskId: task.id,
    group: task.group,
    arm,
    seed,
    transparent: task.transparent,
    width: task.width,
    height: task.height,
    llm: { inputTokens: 0, outputTokens: 0, costUsd: 0, estimatedCostUsd: 0, reportedCostUsd: null },
    checks: [],
    scoredChecks: scoredCheckIds(task.transparent),
  } satisfies Partial<ResultRow>;
}

export function errorOf(err: unknown, timedOut: boolean): NonNullable<ResultRow["error"]> {
  const e = err as { name?: string; code?: string; message?: string };
  return {
    class: timedOut ? "TaskTimeout" : (err instanceof Error ? err.constructor.name : typeof err),
    ...(typeof e?.code === "string" ? { code: e.code } : {}),
    message: String(e?.message ?? err).slice(0, 1000),
    stage: timedOut ? "timeout" : "generate",
  };
}

export async function runArm(arm: Arm, task: EvalTask, seed: number, ctx: ArmContext): Promise<ArmOutput> {
  await mkdir(ctx.workDir, { recursive: true });
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort(new Error(`task timed out after ${ctx.timeoutMs} ms`));
  }, ctx.timeoutMs);
  const started = Date.now();
  try {
    return arm === "raw" ? await runRaw(task, seed, ctx, controller.signal, started) : await runLoop(arm, task, seed, ctx, controller.signal, started);
  } catch (err) {
    return {
      row: { ...baseRow(task, arm, seed), status: "error", error: errorOf(err, timedOut), rounds: 0, wallMs: Date.now() - started, comfyMs: 0 },
    };
  } finally {
    clearTimeout(timer);
  }
}

async function runRaw(task: EvalTask, seed: number, ctx: ArmContext, signal: AbortSignal, started: number): Promise<ArmOutput> {
  const req = request(task, seed);
  const info = await ctx.comfy.getObjectInfo();
  const checkpoint = chooseCheckpoint(info);
  const family = detectFamily(checkpoint);
  const gen = planGenerationSize(family, task.width, task.height);
  const sampling = { ...defaultSampling(family, info), seed };
  const keyColor = task.transparent ? RAW_KEY_COLOR : undefined;
  const doc = buildTxt2Img({
    checkpoint,
    positive: composeKeyedPrompt(task.prompt, keyColor),
    negative: keyColor ? KEY_NEGATIVE : "",
    width: gen.width,
    height: gen.height,
    filenamePrefix: "comfy-agent-eval/raw",
    ...sampling,
  });
  const t = Date.now();
  const run = await ctx.comfy.run(doc.graph, { timeoutMs: ctx.timeoutMs, signal });
  const comfyMs = Date.now() - t;
  const rawPng = run.images.at(-1)!.bytes;
  await writeFile(path.join(ctx.workDir, "round-1-raw.png"), rawPng);
  const post = postProcess(decodePng(rawPng), req, keyColor);
  const imagePath = path.join(ctx.workDir, "asset.png");
  await writeFile(imagePath, encodePng(post.image));
  const checks = deterministicChecks(post.image, req, post.key);
  const tracePath = path.join(ctx.workDir, "trace.json");
  const wallMs = Date.now() - started;
  await writeFile(
    tracePath,
    JSON.stringify({ arm: "raw", request: req, checkpoint, generationSize: gen, sampling, keyColor: post.key, graph: doc.graph, promptId: run.promptId, comfyMs, wallMs, checks }, null, 2),
  );
  return {
    row: { ...baseRow(task, "raw", seed), status: "ok", rounds: 1, stopReason: "single-shot", wallMs, comfyMs, promptIds: [run.promptId], checkpoint, checks, imagePath, tracePath },
    image: post.image,
  };
}

async function runLoop(arm: "plan" | "loop", task: EvalTask, seed: number, ctx: ArmContext, signal: AbortSignal, started: number): Promise<ArmOutput> {
  const req = request(task, seed);
  const res = await generateAsset(req, {
    comfy: ctx.comfy,
    llm: ctx.llm,
    plannerModel: ctx.plannerModel,
    criticModel: ctx.criticModel,
    plannerShape: ctx.shape,
    criticShape: ctx.shape,
    ...(ctx.plannerReasoningEffort ? { plannerReasoningEffort: ctx.plannerReasoningEffort } : {}),
    ...(ctx.criticReasoningEffort ? { criticReasoningEffort: ctx.criticReasoningEffort } : {}),
    // Only the round count differs from the loop's defaults; time and cost limits stay as shipped.
    limits: { maxRounds: arm === "plan" ? 1 : ctx.loopMaxRounds },
    workDir: ctx.workDir,
    runTimeoutMs: ctx.timeoutMs,
    signal,
    ...(ctx.onProgress ? { onProgress: ctx.onProgress } : {}),
  });
  const image = decodePng(res.png);
  const imagePath = path.join(ctx.workDir, "asset.png");
  await writeFile(imagePath, res.png);
  const best = res.trace.rounds.find((r) => r.round === res.trace.best?.round);
  // Recompute the code checks on the returned image so every arm is checked the same way.
  const key = best?.keyColor;
  const checks = deterministicChecks(image, req, key);
  const t = res.trace.totals;
  return {
    row: {
      ...baseRow(task, arm, seed),
      status: "ok",
      checkpoint: res.trace.server.checkpoint,
      models: { planner: res.trace.models.planner, critic: res.trace.models.critic },
      rounds: res.trace.rounds.length,
      stopReason: res.trace.stopReason,
      wallMs: Date.now() - started,
      comfyMs: res.trace.rounds.reduce((s, r) => s + r.timing.comfyMs, 0),
      promptIds: res.trace.rounds.map((r) => r.generation?.promptId).filter((p): p is string => typeof p === "string"),
      llm: { inputTokens: t.inputTokens, outputTokens: t.outputTokens, costUsd: t.costUsd, estimatedCostUsd: t.estimatedCostUsd, reportedCostUsd: t.reportedCostUsd },
      checks,
      imagePath,
      tracePath: path.join(ctx.workDir, "trace.json"),
    },
    image,
  };
}
