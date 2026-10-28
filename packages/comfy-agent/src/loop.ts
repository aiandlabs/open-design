// The asset loop: plan → generate → deterministic checks → critic → typed
// repairs → repeat, until every check passes or a round, time or cost limit
// is reached. The best image (by checklist score) is returned with its graph
// and a complete per-round trace.

import { randomInt, randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import type { ComfyClient } from "./client.js";
import { ComfyAgentError, GraphEditError } from "./errors.js";
import { applyEdit, describeDoc, parseGraphEdit, type GraphEdit } from "./edits.js";
import { familySpec } from "./families.js";
import {
  availableBaseGraphs,
  buildTxt2Img,
  chooseEditModel,
  chooseModel,
  defaultSampling,
  detectFamily,
  docFromGraph,
  planGenerationSize,
  type GraphDoc,
  type ModelChoice,
  type ModelFamily,
} from "./graphs.js";
import {
  alphaStats,
  chromaKey,
  coverToSize,
  cutoutSheet,
  decodePng,
  encodePng,
  estimateKeyColor,
  fitWithin,
  keyResidue,
  parseHexColor,
  rectMask,
  toHexColor,
  type NormalizedRect,
  type RgbaImage,
} from "./image.js";
import {
  type AiandClient,
  type ApiShape,
  type ContentPart,
  type StructuredResult,
  type TokenUsage,
} from "./llm.js";
import { summarizeCapabilities } from "./object-info.js";
import {
  CRITIC_SCHEMA,
  CRITIC_SYSTEM,
  CRITIC_TOOL,
  KEY_COLORS,
  KEY_NEGATIVE,
  composeKeyedPrompt,
  type PromptStyle,
  normalizeCriticItems,
  stripKeyedPrompt,
  PLAN_SCHEMA,
  PLAN_TOOL,
  PLANNER_SYSTEM,
  REPAIR_SCHEMA,
  REPAIR_SYSTEM,
  REPAIR_TOOL,
} from "./prompts.js";
import type { ApiWorkflow, ObjectInfo } from "./types.js";
import { assertValidGraph } from "./validate.js";

export type AssetRequest = {
  prompt: string;
  width: number;
  height: number;
  transparent?: boolean;
  /** Style and palette notes. */
  style?: string;
  /** Things the image must not contain. */
  negative?: string;
  /**
   * Model to use: a checkpoint or a diffusion model file name (must be
   * installed as a complete set). Default: the strongest installed model.
   */
  checkpoint?: string;
  seed?: number;
};

export type AssetLimits = {
  maxRounds?: number;
  maxTimeMs?: number;
  maxCostUsd?: number;
};

export type AssetLoopOptions = {
  comfy: ComfyClient;
  llm: AiandClient;
  /** Absent: chosen from the ai& catalog. Set: checked against it (exists, has the role's capabilities). */
  plannerModel?: string;
  criticModel?: string;
  /** API shape per role; defaults to the client's shape. */
  plannerShape?: ApiShape;
  criticShape?: ApiShape;
  plannerReasoningEffort?: string;
  criticReasoningEffort?: string;
  limits?: AssetLimits;
  /** Round images are written here. */
  workDir: string;
  /** Map absolute paths to what the trace should record (e.g. project-relative). */
  tracePath?: (absolutePath: string) => string;
  /** Optional starting graph (API format) instead of a built base graph. */
  baseGraph?: ApiWorkflow;
  /** Deadline per ComfyUI run. */
  runTimeoutMs?: number;
  onProgress?: (message: string) => void;
  signal?: AbortSignal;
};

export type ChecklistItem = { id: string; text: string };
export type CheckResult = { id: string; pass: boolean; detail: string };
export type CriticItem = { id: string; pass: boolean; evidence?: string; repair_hint?: string };

export type LlmCallTrace = {
  role: "planner" | "critic";
  model: string;
  shape: ApiShape;
  usage: TokenUsage;
  reportedCostUsd?: number;
  estimatedCostUsd?: number;
  latencyMs: number;
};

export type RoundTrace = {
  round: number;
  startedAt: string;
  durationMs: number;
  timing: { comfyMs: number; postProcessMs: number; criticMs: number; plannerMs: number };
  graph: ApiWorkflow;
  settings: Record<string, unknown>;
  generation?: { promptId: string; width: number; height: number; via: string };
  keyColor?: { requested: string; measured: string; borderMatch: number };
  rawImagePath?: string;
  imagePath?: string;
  checks: CheckResult[];
  critic?: { items: CriticItem[]; overall?: string; call?: LlmCallTrace; retries?: LlmCallTrace[]; raw?: unknown; error?: string };
  score: { passed: number; total: number; ratio: number };
  repair?: {
    rationale?: string;
    proposed: unknown[];
    applied: unknown[];
    rejected: { edit: unknown; error: string }[];
    call?: LlmCallTrace;
    error?: string;
  };
  error?: { code: string; message: string };
};

export type StopReason = "all-pass" | "max-rounds" | "time-limit" | "cost-limit" | "error" | "aborted";

export type AssetTrace = {
  id: string;
  version: 1;
  startedAt: string;
  finishedAt: string;
  request: AssetRequest;
  limits: Required<AssetLimits>;
  models: { planner: string; critic: string; plannerShape: ApiShape; criticShape: ApiShape };
  server: {
    url: string;
    checkpoint: string;
    family: string;
    generationSize: { width: number; height: number };
    /** Model that runs `edit_previous`, when one is installed. */
    editModel?: { family: string; model: string };
  };
  plan: {
    checklist: ChecklistItem[];
    positivePrompt: string;
    negativePrompt: string;
    keyColor?: string;
    notes?: string;
    call?: LlmCallTrace;
  };
  rounds: RoundTrace[];
  best: { round: number; score: RoundTrace["score"]; imagePath?: string } | null;
  stopReason: StopReason;
  totals: {
    durationMs: number;
    inputTokens: number;
    outputTokens: number;
    reportedCostUsd: number | null;
    estimatedCostUsd: number;
    costUsd: number;
  };
};

export type AssetResult = {
  ok: boolean;
  png: Buffer;
  graph: ApiWorkflow;
  trace: AssetTrace;
};

const DEFAULT_LIMITS: Required<AssetLimits> = { maxRounds: 3, maxTimeMs: 15 * 60 * 1000, maxCostUsd: 1 };

type PlannerEdit =
  | GraphEdit
  | { op: "img2img_from_previous"; denoise: number }
  | { op: "inpaint_previous"; rect: NormalizedRect; denoise: number }
  | { op: "edit_previous"; instruction: string }
  | { op: "set_key_color"; hex: string };

type PlanOutput = {
  checklist: ChecklistItem[];
  positive_prompt: string;
  negative_prompt: string;
  key_color?: string;
  steps?: number;
  cfg?: number;
  sampler_name?: string;
  scheduler?: string;
  notes?: string;
};

export async function generateAsset(request: AssetRequest, options: AssetLoopOptions): Promise<AssetResult> {
  validateRequest(request);
  const startedAt = Date.now();
  const limits: Required<AssetLimits> = { ...DEFAULT_LIMITS, ...stripUndefined(options.limits ?? {}) };
  // Resolve both models before touching ComfyUI, so a bad model id fails fast.
  const [plannerModel, criticModel] = await Promise.all([
    options.llm.resolveModel("planner", options.plannerModel),
    options.llm.resolveModel("critic", options.criticModel),
  ]);
  const plannerShape = options.plannerShape ?? options.llm.shape;
  const criticShape = options.criticShape ?? options.llm.shape;
  const log = (m: string) => options.onProgress?.(m);
  const recordPath = (p: string) => (options.tracePath ? options.tracePath(p) : p);
  await mkdir(options.workDir, { recursive: true });

  const info = await options.comfy.getObjectInfo();
  const caps = summarizeCapabilities(info);
  if (!availableBaseGraphs(info).includes("txt2img")) {
    throw new ComfyAgentError("GRAPH_EDIT", "this ComfyUI server cannot run a basic text-to-image graph (missing core nodes or checkpoints)");
  }

  // Starting document: the caller's graph, or a built txt2img base graph with
  // the strongest installed model (SDXL when nothing newer is installed).
  const choice: ModelChoice | undefined = options.baseGraph ? undefined : chooseModel(info, request.checkpoint);
  const checkpoint = choice?.model ?? "";
  const family = choice?.family ?? (options.baseGraph ? docFromGraph(structuredClone(options.baseGraph)).family : detectFamily(request.checkpoint ?? ""));
  const spec = familySpec(family);
  const promptStyle: PromptStyle = spec ? "sentences" : "tags";
  const editModel = chooseEditModel(info);
  const gen = planGenerationSize(family, request.width, request.height);
  const seed = request.seed ?? randomInt(0, 2 ** 32 - 1);
  log(`model ${choice?.label ?? checkpoint ?? "custom graph"} (${family})${editModel ? `; edits with ${editModel.label}` : ""}`);

  // 1. Plan.
  log(`planning with ${plannerModel}`);
  const planCall = await options.llm.structured<PlanOutput>(
    {
      model: plannerModel,
      system: PLANNER_SYSTEM,
      content: [{ type: "text", text: planUserText(request, caps, family, gen, choice, editModel) }],
      schema: PLAN_SCHEMA,
      toolName: PLAN_TOOL,
      maxTokens: 8192,
      ...(options.plannerReasoningEffort ? { reasoningEffort: options.plannerReasoningEffort } : {}),
    },
    plannerShape,
  );
  const plan = normalizePlan(planCall.data, request);
  let keyColor = request.transparent ? plan.key_color ?? "#00FF00" : undefined;
  let subjectPrompt = plan.positive_prompt;
  let subjectNegative = plan.negative_prompt;
  const composePositive = () => composeKeyedPrompt(subjectPrompt, keyColor, promptStyle);
  const composeNegative = () => (keyColor ? `${subjectNegative}, ${KEY_NEGATIVE}` : subjectNegative);

  const sampling = defaultSampling(family, info);
  // Distilled families (Z-Image-Turbo, FLUX.2 klein, Ming-Image) have fixed steps and CFG.
  if (!spec?.distilled) {
    if (plan.steps && plan.steps >= 8 && plan.steps <= 60) sampling.steps = Math.round(plan.steps);
    if (plan.cfg && plan.cfg >= 1 && plan.cfg <= 15) sampling.cfg = plan.cfg;
    if (plan.sampler_name && caps.samplers.includes(plan.sampler_name)) sampling.sampler = plan.sampler_name;
    if (plan.scheduler && caps.schedulers.includes(plan.scheduler)) sampling.scheduler = plan.scheduler;
  }
  sampling.seed = seed;

  const buildBase = (seedValue: number): GraphDoc =>
    buildTxt2Img({
      checkpoint,
      family,
      ...(choice?.textEncoder ? { textEncoder: choice.textEncoder } : {}),
      ...(choice?.vae ? { vae: choice.vae } : {}),
      positive: composePositive(),
      negative: composeNegative(),
      width: gen.width,
      height: gen.height,
      filenamePrefix: "comfy-agent/asset",
      ...sampling,
      seed: seedValue,
    });

  let doc: GraphDoc;
  if (options.baseGraph) {
    doc = docFromGraph(structuredClone(options.baseGraph));
    doc = applyEdit(doc, { op: "set_prompt", text: composePositive() }, info);
    if (doc.roles.negative) doc = applyEdit(doc, { op: "set_negative_prompt", text: composeNegative() }, info);
    if (doc.roles.sampler) doc = applyEdit(doc, { op: "set_seed", seed }, info);
  } else {
    doc = buildBase(seed);
  }
  /** Prompt changes on an edit graph start again from text-to-image with the new prompt. */
  const reprompt = (): GraphDoc => {
    if (doc.kind === "edit" && !options.baseGraph) return buildBase(Number(describeDoc(doc).seed ?? seed));
    let d = applyEdit(doc, { op: "set_prompt", text: composePositive() }, info);
    if (d.roles.negative) d = applyEdit(d, { op: "set_negative_prompt", text: composeNegative() }, info);
    return d;
  };

  const trace: AssetTrace = {
    id: randomUUID(),
    version: 1,
    startedAt: new Date(startedAt).toISOString(),
    finishedAt: "",
    request,
    limits,
    models: { planner: plannerModel, critic: criticModel, plannerShape, criticShape },
    server: {
      url: options.comfy.baseUrl,
      checkpoint: String(describeDoc(doc).checkpoint ?? checkpoint),
      family: doc.family,
      generationSize: { width: gen.width, height: gen.height },
      ...(editModel ? { editModel: { family: editModel.family, model: editModel.model } } : {}),
    },
    plan: {
      checklist: plan.checklist,
      positivePrompt: plan.positive_prompt,
      negativePrompt: plan.negative_prompt,
      ...(keyColor ? { keyColor } : {}),
      ...(plan.notes ? { notes: plan.notes } : {}),
      call: callTrace("planner", planCall),
    },
    rounds: [],
    best: null,
    stopReason: "max-rounds",
    totals: { durationMs: 0, inputTokens: 0, outputTokens: 0, reportedCostUsd: null, estimatedCostUsd: 0, costUsd: 0 },
  };
  addUsage(trace, planCall);

  let best: { round: number; png: Buffer; graph: ApiWorkflow; score: RoundTrace["score"]; imagePath?: string } | null = null;
  let previousRaw: Buffer | null = null;
  const seenGraphs = new Set<string>();

  for (let round = 1; round <= limits.maxRounds; round += 1) {
    if (options.signal?.aborted) {
      trace.stopReason = "aborted";
      break;
    }
    const elapsed = Date.now() - startedAt;
    const avgRound = trace.rounds.length ? trace.rounds.reduce((s, r) => s + r.durationMs, 0) / trace.rounds.length : 0;
    if (round > 1 && elapsed + avgRound > limits.maxTimeMs) {
      trace.stopReason = "time-limit";
      break;
    }
    if (round > 1 && trace.totals.costUsd >= limits.maxCostUsd) {
      trace.stopReason = "cost-limit";
      break;
    }

    // Never queue a graph identical to an earlier round: change the seed instead.
    let graphKey = JSON.stringify(doc.graph);
    if (seenGraphs.has(graphKey) && doc.roles.sampler) {
      const cur = Number(describeDoc(doc).seed ?? 0);
      doc = applyEdit(doc, { op: "set_seed", seed: (cur + 7_919_993) % 2 ** 32 }, info);
      graphKey = JSON.stringify(doc.graph);
    }
    seenGraphs.add(graphKey);

    const roundStart = Date.now();
    const rt: RoundTrace = {
      round,
      startedAt: new Date(roundStart).toISOString(),
      durationMs: 0,
      timing: { comfyMs: 0, postProcessMs: 0, criticMs: 0, plannerMs: 0 },
      graph: structuredClone(doc.graph),
      settings: describeDoc(doc),
      checks: [],
      score: { passed: 0, total: 0, ratio: 0 },
    };
    trace.rounds.push(rt);
    log(`round ${round}: generating`);

    // 2. Run.
    let rawPng: Buffer;
    try {
      assertValidGraph(doc.graph, info);
      const t = Date.now();
      const run = await options.comfy.run(doc.graph, {
        validate: false,
        ...(options.runTimeoutMs ? { timeoutMs: options.runTimeoutMs } : {}),
        ...(options.signal ? { signal: options.signal } : {}),
        onProgress: (e) => {
          if (e.type === "progress" && e.max > 0 && (e.value === e.max || e.value % 5 === 0)) log(`round ${round}: step ${e.value}/${e.max}`);
        },
      });
      rt.timing.comfyMs = Date.now() - t;
      const image = run.images.at(-1)!;
      rawPng = image.bytes;
      const raw = decodePng(rawPng);
      rt.generation = { promptId: run.promptId, width: raw.width, height: raw.height, via: run.via };
    } catch (err) {
      rt.error = errorInfo(err);
      rt.durationMs = Date.now() - roundStart;
      if (!best) throw err; // nothing to fall back to
      trace.stopReason = options.signal?.aborted ? "aborted" : "error";
      break;
    }
    const rawPath = path.join(options.workDir, `round-${round}-raw.png`);
    await writeFile(rawPath, rawPng);
    rt.rawImagePath = recordPath(rawPath);
    previousRaw = rawPng;

    // 3. Post-process in code and run deterministic checks.
    const tp = Date.now();
    const post = postProcess(decodePng(rawPng), request, keyColor);
    const finalPng = encodePng(post.image);
    const finalPath = path.join(options.workDir, `round-${round}.png`);
    await writeFile(finalPath, finalPng);
    rt.imagePath = recordPath(finalPath);
    if (post.key) rt.keyColor = post.key;
    rt.checks = deterministicChecks(post.image, request, post.key);
    rt.timing.postProcessMs = Date.now() - tp;

    // 4. Critic.
    const tc = Date.now();
    const criticView = encodePng(fitWithin(request.transparent ? cutoutSheet(post.image) : post.image, request.transparent ? 1024 : 768));
    try {
      log(`round ${round}: grading with ${criticModel}`);
      const grade = async (reminder: string) => {
        const call = await options.llm.structured<Record<string, unknown>>(
          {
            model: criticModel,
            system: CRITIC_SYSTEM,
            content: [
              { type: "image", mime: "image/png", base64: criticView.toString("base64") },
              { type: "text", text: criticUserText(plan.checklist, request) + reminder },
            ],
            schema: CRITIC_SCHEMA,
            toolName: CRITIC_TOOL,
            maxTokens: 4096,
            ...(options.criticReasoningEffort ? { reasoningEffort: options.criticReasoningEffort } : {}),
          },
          criticShape,
        );
        addUsage(trace, call);
        return { call, graded: normalizeCriticItems(call.data) };
      };
      let attempt = await grade("");
      const calls = [callTrace("critic", attempt.call)];
      const missing = (g: CriticItem[]) => plan.checklist.filter((c) => !g.some((it) => it.id === c.id));
      let raw: unknown;
      if (missing(attempt.graded).length > 0) {
        // Keep what the model actually sent, then ask once more for every item.
        raw = attempt.call.data;
        const retry = await grade(`\nReturn an "items" array with exactly one entry for each of these ids: ${plan.checklist.map((c) => c.id).join(", ")}. "pass" must be a boolean.`);
        calls.push(callTrace("critic", retry.call));
        if (missing(retry.graded).length < missing(attempt.graded).length) attempt = retry;
      }
      const overall = typeof attempt.call.data.overall === "string" ? attempt.call.data.overall : undefined;
      rt.critic = {
        items: alignCritic(plan.checklist, attempt.graded),
        call: calls[0]!,
        ...(calls.length > 1 ? { retries: calls.slice(1) } : {}),
        ...(raw !== undefined ? { raw } : {}),
        ...(overall ? { overall } : {}),
      };
    } catch (err) {
      rt.critic = { items: plan.checklist.map((c) => ({ id: c.id, pass: false, evidence: "critic unavailable" })), error: errorInfo(err).message };
    }
    rt.timing.criticMs = Date.now() - tc;

    const passed = rt.checks.filter((c) => c.pass).length + rt.critic.items.filter((c) => c.pass).length;
    const total = rt.checks.length + rt.critic.items.length;
    rt.score = { passed, total, ratio: total ? passed / total : 0 };
    if (!best || rt.score.ratio > best.score.ratio) {
      best = { round, png: finalPng, graph: structuredClone(doc.graph), score: rt.score, imagePath: rt.imagePath };
    }
    const allPass = !rt.critic.error && passed === total;
    if (allPass) {
      rt.durationMs = Date.now() - roundStart;
      trace.stopReason = "all-pass";
      break;
    }
    if (round === limits.maxRounds) {
      rt.durationMs = Date.now() - roundStart;
      trace.stopReason = "max-rounds";
      break;
    }
    if (trace.totals.costUsd >= limits.maxCostUsd) {
      rt.durationMs = Date.now() - roundStart;
      trace.stopReason = "cost-limit";
      break;
    }

    // 5. Repair.
    const tr = Date.now();
    const repair: NonNullable<RoundTrace["repair"]> = { proposed: [], applied: [], rejected: [] };
    rt.repair = repair;
    try {
      log(`round ${round}: planning repairs`);
      const call = await options.llm.structured<{ rationale?: string; edits?: unknown[] }>(
        {
          model: plannerModel,
          system: REPAIR_SYSTEM,
          content: [
            { type: "image", mime: "image/png", base64: criticView.toString("base64") },
            { type: "text", text: repairUserText(request, plan.checklist, doc, rt, caps, trace.rounds, keyColor) },
          ],
          schema: REPAIR_SCHEMA,
          toolName: REPAIR_TOOL,
          maxTokens: 8192,
          ...(options.plannerReasoningEffort ? { reasoningEffort: options.plannerReasoningEffort } : {}),
        },
        plannerShape,
      );
      addUsage(trace, call);
      repair.call = callTrace("planner", call);
      if (call.data.rationale) repair.rationale = call.data.rationale;
      repair.proposed = Array.isArray(call.data.edits) ? call.data.edits : [];
    } catch (err) {
      repair.error = errorInfo(err).message;
    }

    for (const raw of repair.proposed) {
      try {
        const edit = parsePlannerEdit(raw);
        if (FIXED_FOR_DISTILLED.has(edit.op) && (spec?.distilled || doc.kind === "edit")) {
          throw new GraphEditError(edit.op, `${spec?.label ?? doc.family} is distilled: steps, cfg, sampler and scheduler are fixed`);
        }
        switch (edit.op) {
          case "set_prompt":
            subjectPrompt = stripKeyedPrompt(edit.text);
            doc = reprompt();
            break;
          case "set_negative_prompt":
            subjectNegative = edit.text.replace(`, ${KEY_NEGATIVE}`, "");
            if (doc.kind === "edit" || !doc.roles.negative) {
              throw new GraphEditError(edit.op, `${spec?.label ?? doc.family} ignores the negative prompt; state the requirement in set_prompt`);
            }
            doc = applyEdit(doc, { op: "set_negative_prompt", text: composeNegative() }, info);
            break;
          case "set_key_color":
            if (!request.transparent) throw new GraphEditError(edit.op, "the request does not need transparency");
            keyColor = edit.hex;
            doc = reprompt();
            break;
          case "edit_previous": {
            if (!previousRaw) throw new GraphEditError(edit.op, "no previous image");
            if (!editModel) throw new GraphEditError(edit.op, "no instruction-editing model is installed; use img2img_from_previous or inpaint_previous");
            const up = await options.comfy.uploadImage(previousRaw, `comfy-agent-${trace.id.slice(0, 8)}-r${round}-edit.png`);
            const instruction = keyColor
              ? `${edit.instruction.replace(/\s+$/, "").replace(/([^.!?])$/, "$1.")} Keep the plain flat ${KEY_COLORS[keyColor] ?? "key-colour"} background unchanged.`
              : edit.instruction;
            doc = applyEdit(doc, { op: "edit_with_instruction", image: up.name, instruction }, info);
            break;
          }
          case "img2img_from_previous": {
            if (!previousRaw) throw new GraphEditError(edit.op, "no previous image");
            const up = await options.comfy.uploadImage(previousRaw, `comfy-agent-${trace.id.slice(0, 8)}-r${round}.png`);
            doc = applyEdit(doc, { op: "use_img2img", image: up.name, denoise: edit.denoise }, info);
            break;
          }
          case "inpaint_previous": {
            if (!previousRaw) throw new GraphEditError(edit.op, "no previous image");
            const prev = decodePng(previousRaw);
            const up = await options.comfy.uploadImage(previousRaw, `comfy-agent-${trace.id.slice(0, 8)}-r${round}.png`);
            const mask = await options.comfy.uploadImage(
              encodePng(rectMask(prev.width, prev.height, edit.rect, Math.round(Math.min(prev.width, prev.height) * 0.03))),
              `comfy-agent-${trace.id.slice(0, 8)}-r${round}-mask.png`,
            );
            doc = applyEdit(doc, { op: "use_inpaint", image: up.name, mask: mask.name, denoise: edit.denoise }, info);
            break;
          }
          default:
            doc = applyEdit(doc, edit, info);
        }
        repair.applied.push(edit);
      } catch (err) {
        repair.rejected.push({ edit: raw, error: errorInfo(err).message });
      }
    }
    rt.timing.plannerMs = Date.now() - tr;
    rt.durationMs = Date.now() - roundStart;
  }

  const finished = Date.now();
  trace.finishedAt = new Date(finished).toISOString();
  trace.totals.durationMs = finished - startedAt;
  if (!best) throw new ComfyAgentError("COMFY_MISSING_OUTPUT", "the asset loop produced no image");
  trace.best = { round: best.round, score: best.score, ...(best.imagePath ? { imagePath: best.imagePath } : {}) };
  const tracePathAbs = path.join(options.workDir, "trace.json");
  await writeFile(tracePathAbs, JSON.stringify(trace, null, 2));
  return { ok: trace.stopReason === "all-pass", png: best.png, graph: best.graph, trace };
}

// ---------------------------------------------------------------------------

function validateRequest(r: AssetRequest): void {
  if (!r.prompt || !r.prompt.trim()) throw new ComfyAgentError("GRAPH_EDIT", "request.prompt is required");
  for (const k of ["width", "height"] as const) {
    if (!Number.isInteger(r[k]) || r[k] < 16 || r[k] > 8192) throw new ComfyAgentError("GRAPH_EDIT", `request.${k} must be an integer in [16, 8192]`);
  }
}

function stripUndefined<T extends object>(o: T): Partial<T> {
  return Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined)) as Partial<T>;
}

export function postProcess(
  raw: RgbaImage,
  request: AssetRequest,
  keyColor: string | undefined,
): { image: RgbaImage; key?: { requested: string; measured: string; borderMatch: number } } {
  if (request.transparent && keyColor) {
    const nominal = parseHexColor(keyColor);
    const est = estimateKeyColor(raw, nominal);
    const keyed = chromaKey(raw, est.color, { tolerance: 0.1, softness: 0.1, connectedOnly: true, featherPx: 1, despill: true });
    return {
      image: coverToSize(keyed, request.width, request.height),
      key: { requested: keyColor, measured: toHexColor(est.color), borderMatch: Number(est.borderMatch.toFixed(3)) },
    };
  }
  return { image: coverToSize(raw, request.width, request.height) };
}

// A clean cut-out keeps partial alpha to a thin edge band and has almost no
// visible key-coloured pixels. Calibrated on six live SDXL lantern rounds:
// the bad rounds measured 1.7–30% tinted or 12–45% haze; the clean one
// measured 0.4% and 4.7%.
const KEY_RESIDUE_MAX_TINTED = 0.01;
const KEY_RESIDUE_MAX_HAZE = 0.08;

export function deterministicChecks(
  img: RgbaImage,
  request: AssetRequest,
  key?: { requested: string; measured: string; borderMatch: number },
): CheckResult[] {
  const checks: CheckResult[] = [
    {
      id: "size",
      pass: img.width === request.width && img.height === request.height,
      detail: `${img.width}x${img.height} (requested ${request.width}x${request.height})`,
    },
  ];
  if (request.transparent) {
    const a = alphaStats(img);
    checks.push({
      id: "alpha",
      pass: a.transparent >= 0.05 && a.borderTransparent >= 0.6,
      detail: `transparent ${(a.transparent * 100).toFixed(1)}%, border transparent ${(a.borderTransparent * 100).toFixed(1)}%`,
    });
    if (key) {
      checks.push({
        id: "flat_background",
        pass: key.borderMatch >= 0.6,
        detail: `${(key.borderMatch * 100).toFixed(0)}% of border pixels near key ${key.requested} (measured ${key.measured})`,
      });
      const residue = keyResidue(img, parseHexColor(key.measured));
      checks.push({
        id: "key_residue",
        pass: residue.keyTinted <= KEY_RESIDUE_MAX_TINTED && residue.haze <= KEY_RESIDUE_MAX_HAZE,
        detail:
          `${(residue.keyTinted * 100).toFixed(1)}% of visible pixels still near the key colour (max ${KEY_RESIDUE_MAX_TINTED * 100}%), ` +
          `${(residue.haze * 100).toFixed(1)}% partly transparent (max ${KEY_RESIDUE_MAX_HAZE * 100}%); ` +
          "high values mean a shadow, glow or haze was painted on the background",
      });
    }
  }
  return checks;
}

function normalizePlan(raw: PlanOutput, request: AssetRequest): PlanOutput {
  const items = Array.isArray(raw.checklist) ? raw.checklist : [];
  const seen = new Set<string>();
  const checklist: ChecklistItem[] = [];
  for (const [i, item] of items.entries()) {
    if (!item || typeof item.text !== "string" || !item.text.trim()) continue;
    let id = (typeof item.id === "string" && item.id.trim() ? item.id.trim() : `item${i + 1}`).replace(/\s+/g, "_");
    while (seen.has(id)) id = `${id}_${i}`;
    seen.add(id);
    checklist.push({ id, text: item.text.trim() });
  }
  if (checklist.length === 0) checklist.push({ id: "subject", text: `The image shows: ${request.prompt}` });
  const key = typeof raw.key_color === "string" ? raw.key_color.toUpperCase() : undefined;
  return {
    ...raw,
    checklist: checklist.slice(0, 8),
    positive_prompt: typeof raw.positive_prompt === "string" && raw.positive_prompt.trim() ? raw.positive_prompt.trim() : request.prompt,
    negative_prompt: [typeof raw.negative_prompt === "string" ? raw.negative_prompt.trim() : "", request.negative?.trim() ?? ""].filter(Boolean).join(", "),
    ...(key && KEY_COLORS[key] ? { key_color: key } : { key_color: undefined }),
  };
}

/** Settings a distilled model (or an edit graph) runs with fixed values. */
const FIXED_FOR_DISTILLED = new Set<string>(["set_steps", "set_cfg", "set_sampler", "set_scheduler", "set_denoise"]);

function parsePlannerEdit(raw: unknown): PlannerEdit {
  const r = (raw ?? {}) as Record<string, unknown>;
  if (r.op === "edit_previous") {
    const instruction = typeof r.instruction === "string" ? r.instruction.trim() : typeof r.text === "string" ? r.text.trim() : "";
    if (!instruction) throw new GraphEditError("edit_previous", "instruction must be a non-empty string");
    return { op: "edit_previous", instruction };
  }
  if (r.op === "edit_with_instruction") {
    throw new GraphEditError("edit_with_instruction", "use edit_previous instead");
  }
  if (r.op === "img2img_from_previous") {
    const d = Number(r.denoise ?? 0.55);
    if (!(d > 0 && d <= 1)) throw new GraphEditError("img2img_from_previous", "denoise must be in (0, 1]");
    return { op: "img2img_from_previous", denoise: d };
  }
  if (r.op === "inpaint_previous") {
    const rect = { x: Number(r.x), y: Number(r.y), width: Number(r.width), height: Number(r.height) };
    if (![rect.x, rect.y, rect.width, rect.height].every((v) => Number.isFinite(v) && v >= 0 && v <= 1) || rect.width <= 0 || rect.height <= 0) {
      throw new GraphEditError("inpaint_previous", "x, y, width, height must be fractions in [0, 1] with a non-empty area");
    }
    const d = Number(r.denoise ?? 0.75);
    if (!(d > 0 && d <= 1)) throw new GraphEditError("inpaint_previous", "denoise must be in (0, 1]");
    return { op: "inpaint_previous", rect, denoise: d };
  }
  if (r.op === "set_key_color") {
    const hex = String(r.hex ?? "").toUpperCase();
    if (!KEY_COLORS[hex]) throw new GraphEditError("set_key_color", `hex must be one of ${Object.keys(KEY_COLORS).join(", ")}`);
    return { op: "set_key_color", hex };
  }
  if (r.op === "set_seed" && (r.seed === undefined || r.seed === null || r.seed === "" || !Number.isFinite(Number(r.seed)))) {
    // Planners sometimes ask for "a new seed" without a value.
    return { op: "set_seed", seed: randomInt(0, 2 ** 32 - 1) };
  }
    if (r.op === "use_img2img" || r.op === "use_inpaint") {
    throw new GraphEditError(String(r.op), "use img2img_from_previous / inpaint_previous instead");
  }
  return parseGraphEdit(raw);
}

function alignCritic(checklist: ChecklistItem[], items: CriticItem[]): CriticItem[] {
  // Exactly one grade per checklist item, in checklist order; missing grades fail.
  return checklist.map((c, i) => {
    const found = items.find((it) => it && it.id === c.id) ?? (items.length === checklist.length ? items[i] : undefined);
    if (!found || typeof found.pass !== "boolean") return { id: c.id, pass: false, evidence: "not graded" };
    return {
      id: c.id,
      pass: found.pass,
      ...(found.evidence ? { evidence: String(found.evidence) } : {}),
      ...(found.repair_hint ? { repair_hint: String(found.repair_hint) } : {}),
    };
  });
}

function callTrace(role: "planner" | "critic", r: StructuredResult<unknown>): LlmCallTrace {
  return {
    role,
    model: r.model,
    shape: r.shape,
    usage: r.usage,
    latencyMs: r.latencyMs,
    ...(r.reportedCostUsd === undefined ? {} : { reportedCostUsd: r.reportedCostUsd }),
    ...(r.estimatedCostUsd === undefined ? {} : { estimatedCostUsd: r.estimatedCostUsd }),
  };
}

function addUsage(trace: AssetTrace, r: StructuredResult<unknown>): void {
  trace.totals.inputTokens += r.usage.inputTokens;
  trace.totals.outputTokens += r.usage.outputTokens;
  if (r.reportedCostUsd !== undefined) trace.totals.reportedCostUsd = (trace.totals.reportedCostUsd ?? 0) + r.reportedCostUsd;
  trace.totals.estimatedCostUsd += r.estimatedCostUsd ?? 0;
  trace.totals.costUsd += r.reportedCostUsd ?? r.estimatedCostUsd ?? 0;
}

function errorInfo(err: unknown): { code: string; message: string } {
  if (err instanceof ComfyAgentError) return { code: err.code, message: err.message };
  return { code: "ERROR", message: err instanceof Error ? err.message : String(err) };
}

function planUserText(
  request: AssetRequest,
  caps: ReturnType<typeof summarizeCapabilities>,
  family: ModelFamily,
  gen: { width: number; height: number },
  choice: ModelChoice | undefined,
  editModel: ModelChoice | undefined,
): string {
  const spec = familySpec(family);
  return [
    `Request: ${request.prompt}`,
    `Output size: ${request.width}x${request.height} px (generated at ${gen.width}x${gen.height}, then cropped/resized in code).`,
    `Transparent background required: ${request.transparent ? "yes — choose key_color" : "no"}`,
    request.style ? `Style and palette notes: ${request.style}` : "",
    request.negative ? `Must not contain: ${request.negative}` : "",
    spec
      ? `Model family: ${family} (${choice?.label ?? spec.label}, distilled, language-model text encoder). Write positive_prompt as plain sentences; steps, cfg, sampler and scheduler are fixed.`
      : `Model family: ${family}. Available samplers: ${caps.samplers.slice(0, 25).join(", ")}. Schedulers: ${caps.schedulers.join(", ")}.`,
    editModel ? `Instruction editing is available for repairs (${editModel.label}).` : "",
    `Allowed key colours: ${Object.entries(KEY_COLORS).map(([h, n]) => `${h} (${n})`).join(", ")}.`,
    `Call ${PLAN_TOOL}.`,
  ].filter(Boolean).join("\n");
}

function criticUserText(checklist: ChecklistItem[], request: AssetRequest): string {
  return [
    `Original request: ${request.prompt}`,
    request.transparent
      ? "This is a cut-out asset, shown twice: on white (left) and on near-black (right). Those backgrounds are transparent. " +
        "A visible shadow, glow, tint, haze or coloured fringe around the subject on either side is a defect: fail any item it affects."
      : "",
    "Checklist:",
    ...checklist.map((c) => `- ${c.id}: ${c.text}`),
    `Grade every item by id and call ${CRITIC_TOOL}.`,
  ].filter(Boolean).join("\n");
}

function repairUserText(
  request: AssetRequest,
  checklist: ChecklistItem[],
  doc: GraphDoc,
  rt: RoundTrace,
  caps: ReturnType<typeof summarizeCapabilities>,
  rounds: RoundTrace[],
  keyColor: string | undefined,
): string {
  const failing = rt.critic?.items.filter((i) => !i.pass) ?? [];
  const history = rounds.slice(0, -1).map((r) => ({
    round: r.round,
    score: `${r.score.passed}/${r.score.total}`,
    applied: r.repair?.applied ?? [],
    rejected: r.repair?.rejected.map((x) => x.error) ?? [],
  }));
  return [
    `Request: ${request.prompt} (${request.width}x${request.height}${request.transparent ? `, transparent via key colour ${keyColor}` : ""})`,
    request.style ? `Style notes: ${request.style}` : "",
    `Checklist: ${JSON.stringify(checklist)}`,
    `Current settings: ${JSON.stringify(describeDoc(doc))}`,
    `Code checks: ${JSON.stringify(rt.checks)}`,
    `Critic grades: ${JSON.stringify(rt.critic?.items ?? [])}`,
    failing.length ? `Failing: ${failing.map((f) => `${f.id} — ${f.repair_hint ?? f.evidence ?? ""}`).join("; ")}` : "Only code checks are failing.",
    history.length ? `Earlier rounds: ${JSON.stringify(history)}` : "",
    `Model family: ${doc.family}${doc.kind === "edit" ? " (the current graph is an instruction edit of the previous image)" : ""}.`,
    caps.editModel
      ? `Instruction editing: available (${caps.editModel.family}, ${caps.editModel.model}); use edit_previous for detail fixes.`
      : "Instruction editing: not available on this server.",
    `Available: samplers ${caps.samplers.slice(0, 25).join(", ")}; schedulers ${caps.schedulers.join(", ")}; checkpoints ${caps.checkpoints.join(", ")}; LoRAs ${caps.loras.join(", ") || "none"}; upscale models ${caps.upscaleModels.join(", ") || "none"}.`,
    "When you change the prompt with set_prompt, give the full new subject prompt (the background phrase is added automatically).",
    `Call ${REPAIR_TOOL}.`,
  ].filter(Boolean).join("\n");
}

/** Type re-exports for callers composing their own loops. */
export type { ContentPart, GraphEdit, ObjectInfo };
