// The arms. Each produces one final image for a task and seed:
//   raw          the user's prompt straight into the default text-to-image graph
//   plan         the planner plus one generation (maxRounds 1)
//   loop         the full asset loop
//   loop-chroma  the loop, cutting out with chroma keying instead of matting
//   loop-noref   the loop without shape references (icons)
//   loop-noset   the loop without the set's earlier icons as style references
// Generation size, checkpoint, sampling defaults, cut-out method,
// post-processing and code checks are shared with the loop so the arms differ
// only in what they test.

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import type { ComfyClient } from "../../src/client.js";
import { addBackgroundRemoval, buildTxt2Img, chooseModel, defaultSampling, planGenerationSize } from "../../src/graphs.js";
import { isSplitFamily } from "../../src/families.js";
import { decodePng, encodePng, type RgbaImage } from "../../src/image.js";
import type { AiandClient, ApiShape } from "../../src/llm.js";
import { deterministicChecks, generateAsset, postProcess, type AssetRequest, type ReferenceProvider, type SetReference } from "../../src/loop.js";
import { chooseBackgroundRemovalModel } from "../../src/object-info.js";
import { composeKeyedPrompt, KEY_NEGATIVE } from "../../src/prompts.js";
import { scoredCheckIds, type Arm, type ResultRow } from "./results.js";
import type { EvalTask } from "./tasks.js";

/** Key colour arm A uses for transparent tasks when no matting model is installed (the loop's own fallback). */
export const RAW_KEY_COLOR = "#00FF00";
/** Backdrop arm A renders transparent tasks on when it cuts them out with the matting model. */
export const RAW_MATTE_BACKDROP = "#FFFFFF";

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
  /** Shape references for icon tasks (absent: none; the loop-noref arm never uses them). */
  library?: ReferenceProvider;
  /** Set tasks: the set's earlier icons from the same arm and seed (the loop-noset arm never uses them). */
  setReferences?: SetReference[];
  onProgress?: (message: string) => void;
};

export type ArmOutput = {
  row: Omit<ResultRow, "judge" | "judge2" | "judgeViewPath" | "finishedAt">;
  image?: RgbaImage;
};

function request(task: EvalTask, seed: number, arm: Arm): AssetRequest {
  return {
    prompt: task.prompt,
    width: task.width,
    height: task.height,
    transparent: task.transparent,
    seed,
    ...(arm === "loop-chroma" ? { cutout: "chroma-key" as const } : {}),
    ...(task.group === "icon" ? { kind: "icon" as const } : {}),
  };
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
    ...(task.set ? { set: task.set } : {}),
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
  const req = request(task, seed, "raw");
  const info = await ctx.comfy.getObjectInfo();
  // The same model the plan and loop arms use (the server default), so the
  // arms differ only in planning and repair, not in the image model.
  const choice = chooseModel(info);
  const checkpoint = choice.model;
  const family = choice.family;
  const gen = planGenerationSize(family, task.width, task.height);
  const sampling = { ...defaultSampling(family, info), seed };
  // The same cut-out method as the loop's default: matting when installed.
  const matteModel = task.transparent ? chooseBackgroundRemovalModel(info) : undefined;
  const keyColor = task.transparent ? (matteModel ? RAW_MATTE_BACKDROP : RAW_KEY_COLOR) : undefined;
  const doc = buildTxt2Img({
    checkpoint,
    family,
    ...(choice.textEncoder ? { textEncoder: choice.textEncoder } : {}),
    ...(choice.vae ? { vae: choice.vae } : {}),
    positive: composeKeyedPrompt(task.prompt, keyColor, isSplitFamily(family) ? "sentences" : "tags"),
    negative: keyColor ? KEY_NEGATIVE : "",
    width: gen.width,
    height: gen.height,
    filenamePrefix: "comfy-agent-eval/raw",
    ...sampling,
  });
  const graph = matteModel ? addBackgroundRemoval(doc.graph, matteModel) : doc.graph;
  const t = Date.now();
  const run = await ctx.comfy.run(graph, { timeoutMs: ctx.timeoutMs, signal });
  const comfyMs = Date.now() - t;
  const out = decodePng(run.images.at(-1)!.bytes);
  // With matting the output carries the matte as alpha; the raw render is its opaque RGB.
  const opaque = matteModel ? { ...out, data: out.data.map((v, i) => (i % 4 === 3 ? 255 : v)) } : out;
  await writeFile(path.join(ctx.workDir, "round-1-raw.png"), encodePng(opaque));
  const post = postProcess(opaque, req, keyColor, matteModel ? { matted: out } : {});
  const imagePath = path.join(ctx.workDir, "asset.png");
  await writeFile(imagePath, encodePng(post.image));
  const checks = deterministicChecks(post.image, req, post.key, post.cutout);
  const tracePath = path.join(ctx.workDir, "trace.json");
  const wallMs = Date.now() - started;
  await writeFile(
    tracePath,
    JSON.stringify({ arm: "raw", request: req, checkpoint, generationSize: gen, sampling, keyColor: post.key, cutout: post.cutout, graph, promptId: run.promptId, comfyMs, wallMs, checks }, null, 2),
  );
  const cutout = post.cutout?.method;
  return {
    row: {
      ...baseRow(task, "raw", seed),
      status: "ok",
      rounds: 1,
      stopReason: "single-shot",
      wallMs,
      comfyMs,
      promptIds: [run.promptId],
      checkpoint,
      ...(cutout ? { cutout } : {}),
      checks,
      scoredChecks: scoredCheckIds(task.transparent, cutout),
      imagePath,
      tracePath,
    },
    image: post.image,
  };
}

async function runLoop(arm: Exclude<Arm, "raw">, task: EvalTask, seed: number, ctx: ArmContext, signal: AbortSignal, started: number): Promise<ArmOutput> {
  const req = request(task, seed, arm);
  const library = arm === "loop-noref" ? undefined : ctx.library;
  const set = arm === "loop-noset" ? [] : (ctx.setReferences ?? []);
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
    ...(library || set.length ? { references: { ...(library ? { library } : {}), set } } : {}),
    signal,
    ...(ctx.onProgress ? { onProgress: ctx.onProgress } : {}),
  });
  const image = decodePng(res.png);
  const imagePath = path.join(ctx.workDir, "asset.png");
  await writeFile(imagePath, res.png);
  const best = res.trace.rounds.find((r) => r.round === res.trace.best?.round);
  // Recompute the code checks on the returned image so every arm is checked
  // the same way; the near-copy result comes from the round that was returned.
  const checks = deterministicChecks(image, req, best?.keyColor, best?.cutout);
  const copyCheck = best?.checks.find((c) => c.id === "reference_copy");
  if (copyCheck) checks.push(copyCheck);
  const cutout = best?.cutout?.method;
  const refs = res.trace.references;
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
      ...(cutout ? { cutout } : {}),
      ...(refs ? { references: { shape: refs.shape.map((s) => s.id).filter((id) => !refs.dropped?.some((d) => d.id === id)), set: refs.set.map((s) => s.id) } } : {}),
      checks,
      scoredChecks: scoredCheckIds(task.transparent, cutout, Boolean(copyCheck)),
      imagePath,
      tracePath: path.join(ctx.workDir, "trace.json"),
    },
    image,
  };
}
