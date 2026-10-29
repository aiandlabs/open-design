// The asset loop: plan → generate → deterministic checks → critic → typed
// repairs → repeat, until every check passes or a round, time or cost limit
// is reached. The best image (by checklist score) is returned with its graph
// and a complete per-round trace.

import { randomInt, randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import type { ComfyClient } from "./client.js";
import { ComfyAgentError, ComfyTimeoutError, GraphEditError } from "./errors.js";
import { applyEdit, describeDoc, parseGraphEdit, type GraphEdit } from "./edits.js";
import { familySpec } from "./families.js";
import {
  REFERENCE_FAMILIES,
  addBackgroundRemoval,
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
  NEAR_COPY_MIN_IOU,
  NEAR_COPY_MIN_STRUCTURE,
  alphaStats,
  chromaKey,
  compositeOver,
  contactSheet,
  coverToSize,
  cutoutSheet,
  decodePng,
  decontaminateMatte,
  encodePng,
  estimateKeyColor,
  fitWithin,
  keyResidue,
  matteResidue,
  parseHexColor,
  rectMask,
  referenceSimilarity,
  toHexColor,
  type NormalizedRect,
  type ReferenceSimilarity,
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
  MATCH_SCHEMA,
  MATCH_SYSTEM,
  MATCH_TOOL,
  RECOGNISE_SCHEMA,
  RECOGNISE_SYSTEM,
  RECOGNISE_TOOL,
  KEY_COLORS,
  KEY_NEGATIVE,
  MATTE_BACKDROPS,
  backdropName,
  composeKeyedPrompt,
  referenceSteering,
  type PromptStyle,
  normalizeCriticItems,
  stripKeyedPrompt,
  planSchema,
  PLAN_TOOL,
  PLANNER_SYSTEM,
  REPAIR_SCHEMA,
  REPAIR_SYSTEM,
  REPAIR_TOOL,
} from "./prompts.js";
import type { ReferenceAsset, ReferenceCandidate, ReferenceLibrary } from "./references.js";
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
  /**
   * How a transparent asset is cut out. `auto` (default): a matting model
   * (core RemoveBackground node) when the server has one, else chroma keying.
   */
  cutout?: CutoutMode;
  /**
   * What the asset is. `icon` turns on shape references (closest emoji and
   * icon-font symbols) when the loop is given a reference library.
   */
  kind?: "icon";
};

/** An image already accepted for the same set, used as a style reference. */
export type SetReference = { id: string; png: Buffer; label?: string };

/** Finds and renders shape references; {@link ReferenceLibrary} implements it. */
export type ReferenceProvider = Pick<ReferenceLibrary, "search" | "load"> & { indexDigests?: Record<string, string> };

export type ReferenceOptions = {
  /** Shape and concept references for `kind: "icon"` requests. */
  library?: ReferenceProvider;
  /** Accepted images from the same set (for example the project's earlier icons), most relevant first. */
  set?: SetReference[];
  /** Most shape references per asset (default 1). */
  maxShape?: number;
  /** Most set references per asset (default 3). Every reference costs memory on the GPU. */
  maxSet?: number;
};

export type CutoutMode = "auto" | "matte" | "chroma-key";
export type CutoutMethod = "matte" | "chroma-key";

/** How a round's transparent asset was cut out, and the backdrop it was rendered on. */
export type CutoutInfo = {
  method: CutoutMethod;
  /** Requested backdrop or key colour, the colour measured on the image border, and the share of border pixels near it. */
  backdrop: { requested: string; measured: string; borderMatch: number };
};

export type AssetLimits = {
  maxRounds?: number;
  /**
   * Budget for active time: prompt execution, post-processing and LLM calls.
   * Time spent waiting in a shared ComfyUI queue does not count.
   */
  maxTimeMs?: number;
  /**
   * Wall-clock ceiling, queue waits included, so a stuck queue still ends the
   * run (stop reason `queue-timeout`). Default: 4 × maxTimeMs, at least 60 min.
   */
  maxWallTimeMs?: number;
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
  /** Reference images (FLUX.2 [klein] only; ignored, with a note in the trace, for other models). */
  references?: ReferenceOptions;
  /**
   * Blind recognition check (default on): each round, a vision model that is
   * not told the subject names what the image shows, and the round fails when
   * a typical viewer would not recognise the planned subject.
   */
  recognition?: boolean;
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
  /** As the API reported it (X-Cost header or body), in its own currency. */
  reportedCost?: { amount: number; currency: string; source: "x-cost" | "body" };
  estimatedCostUsd?: number;
  latencyMs: number;
};

export type RoundTrace = {
  round: number;
  startedAt: string;
  durationMs: number;
  /** comfyMs includes queueWaitMs, the time the prompt waited behind other prompts. */
  timing: { comfyMs: number; queueWaitMs: number; postProcessMs: number; criticMs: number; plannerMs: number };
  graph: ApiWorkflow;
  settings: Record<string, unknown>;
  generation?: { promptId: string; width: number; height: number; via: string };
  /** Chroma keying only (kept for older trace readers); `cutout` covers both methods. */
  keyColor?: { requested: string; measured: string; borderMatch: number };
  cutout?: CutoutInfo;
  /** How closely the output matches each reference in use (near-copy guard). */
  referenceSimilarity?: ({ id: string } & ReferenceSimilarity)[];
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
  /** Where the round is, for hosts that show the loop live; `done` once the round is over. */
  phase?: "generating" | "grading" | "repairing" | "done";
  /** The node ComfyUI is executing, and the sampler's step progress (from ComfyUI's websocket). */
  runningNode?: string;
  progress?: { nodeId?: string; value: number; max: number };
  /** True while the prompt waits in ComfyUI's queue behind other prompts. */
  queued?: boolean;
};

/**
 * `time-limit`: the active-time budget (maxTimeMs) ran out.
 * `queue-timeout`: the wall-clock ceiling (maxWallTimeMs) ran out, or ComfyUI
 * never started a prompt, because of waiting in a shared queue.
 */
export type StopReason = "all-pass" | "max-rounds" | "time-limit" | "queue-timeout" | "cost-limit" | "error" | "aborted";

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
    /** Transparent assets only: how the subject is cut out. */
    cutout?: { method: CutoutMethod; model?: string };
  };
  plan: {
    checklist: ChecklistItem[];
    /** What the image should show, as a viewer would name it (the request when the planner gave none). */
    subject?: string;
    positivePrompt: string;
    negativePrompt: string;
    keyColor?: string;
    notes?: string;
    call?: LlmCallTrace;
  };
  /** Reference images offered and used, and where each came from. */
  references?: {
    candidates: { id: string; label: string; score: number; source?: string }[];
    shape: { id: string; label: string; source: string; url?: string; why?: string }[];
    set: { id: string; label?: string }[];
    /** The planner's raw answer about shape references, when candidates were listed. */
    answer?: unknown[];
    /** Why references were not used, or why none was chosen. */
    note?: string;
    /** References a repair stopped using, and in which round. */
    dropped?: { id: string; round: number }[];
    indexDigests?: Record<string, string>;
  };
  rounds: RoundTrace[];
  best: { round: number; score: RoundTrace["score"]; imagePath?: string } | null;
  stopReason: StopReason;
  /** Why the loop failed, when it ended without an image. */
  error?: { code: string; message: string };
  totals: {
    durationMs: number;
    /** Sum of the rounds' queue waits; durationMs minus this is the active time. */
    queueWaitMs: number;
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

const DEFAULT_LIMITS: Omit<Required<AssetLimits>, "maxWallTimeMs"> = { maxRounds: 3, maxTimeMs: 15 * 60 * 1000, maxCostUsd: 1 };

function resolveLimits(given: AssetLimits): Required<AssetLimits> {
  const base = { ...DEFAULT_LIMITS, ...stripUndefined(given) };
  return { ...base, maxWallTimeMs: given.maxWallTimeMs ?? Math.max(4 * base.maxTimeMs, 60 * 60 * 1000) };
}

type PlannerEdit =
  | GraphEdit
  | { op: "img2img_from_previous"; denoise: number }
  | { op: "inpaint_previous"; rect: NormalizedRect; denoise: number }
  | { op: "edit_previous"; instruction: string }
  | { op: "set_key_color"; hex: string }
  | { op: "drop_reference"; id: string };

type PlanOutput = {
  subject?: string;
  references?: { id?: unknown; why?: unknown }[];
  reference_note?: string;
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
  const traceId = randomUUID();
  const limits = resolveLimits(options.limits ?? {});
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
  const cutout = request.transparent ? chooseCutout(request.cutout ?? "auto", caps.backgroundRemoval?.model) : undefined;
  const backdrops = cutout?.method === "matte" ? MATTE_BACKDROPS : KEY_COLORS;
  log(
    `model ${choice?.label ?? checkpoint ?? "custom graph"} (${family})${editModel ? `; edits with ${editModel.label}` : ""}` +
      (cutout ? `; cut-out by ${cutout.method === "matte" ? `matting (${cutout.model})` : "chroma keying"}` : ""),
  );

  // 0. References: shape candidates for icons, and the set's accepted images.
  const refOpts = options.references ?? {};
  const refTrace: NonNullable<AssetTrace["references"]> = { candidates: [], shape: [], set: [] };
  const wantShape = request.kind === "icon" && !!refOpts.library;
  let setRefs: { id: string; label?: string; image: RgbaImage }[] = [];
  let candidates: { candidate: ReferenceCandidate; asset: ReferenceAsset; image: RgbaImage }[] = [];
  const referencesSupported = !options.baseGraph && REFERENCE_FAMILIES.includes(family);
  if ((wantShape || refOpts.set?.length) && !referencesSupported) {
    refTrace.note = `${options.baseGraph ? "a custom base graph" : spec?.label ?? family} cannot take reference images; generated without them`;
  }
  if (referencesSupported) {
    setRefs = (refOpts.set ?? []).slice(0, refOpts.maxSet ?? 3).flatMap((s) => {
      try {
        return [{ id: s.id, ...(s.label ? { label: s.label } : {}), image: decodePng(s.png) }];
      } catch {
        return [];
      }
    });
    refTrace.set = setRefs.map((s) => ({ id: s.id, ...(s.label ? { label: s.label } : {}) }));
    if (wantShape) {
      try {
        log("finding shape references");
        const found = await refOpts.library!.search(request.prompt, 4);
        for (const candidate of found) {
          const asset = await refOpts.library!.load(candidate.id, 512).catch(() => undefined);
          if (asset) candidates.push({ candidate, asset, image: decodePng(asset.png) });
        }
        refTrace.candidates = candidates.map(({ candidate, asset }) => ({ id: candidate.id, label: candidate.label, score: Number(candidate.score.toFixed(2)), source: asset.source }));
        if (refOpts.library!.indexDigests && Object.keys(refOpts.library!.indexDigests).length) refTrace.indexDigests = { ...refOpts.library!.indexDigests };
        if (candidates.length === 0) refTrace.note = "no emoji or icon matched the request";
      } catch (err) {
        candidates = [];
        refTrace.note = `reference search failed: ${errorInfo(err).message}`;
      }
    }
  }
  const maxShape = refOpts.maxShape ?? 1;

  // 1. Plan.
  log(`planning with ${plannerModel}`);
  const planContent: ContentPart[] = [{ type: "text", text: planUserText(request, caps, family, gen, choice, editModel, cutout?.method) }];
  if (candidates.length) {
    planContent.push(
      { type: "image", mime: "image/png", base64: encodePng(contactSheet(candidates.map((c) => c.image), 192)).toString("base64") },
      {
        type: "text",
        text:
          `Reference candidates, numbered left to right in the sheet above (pick at most ${maxShape}, or none):\n` +
          candidates.map((c, i) => `${i + 1}. ${c.candidate.id} ${c.candidate.label} (${c.asset.source})`).join("\n"),
      },
    );
  }
  if (setRefs.length) {
    planContent.push(
      { type: "image", mime: "image/png", base64: encodePng(contactSheet(setRefs.map((s) => s.image), 192)).toString("base64") },
      { type: "text", text: `Icons already made for the same set (above): ${setRefs.map((s) => s.label ?? s.id).join(", ")}. This icon must match their style.` },
    );
  }
  const planCall = await options.llm.structured<PlanOutput>(
    {
      model: plannerModel,
      system: PLANNER_SYSTEM,
      content: planContent,
      schema: planSchema(candidates.length > 0),
      toolName: PLAN_TOOL,
      maxTokens: 8192,
      ...(options.plannerReasoningEffort ? { reasoningEffort: options.plannerReasoningEffort } : {}),
    },
    plannerShape,
  );
  const plan = normalizePlan(planCall.data, request, backdrops);
  if (setRefs.length && !plan.checklist.some((c) => c.id === SET_CONSISTENCY_ITEM.id)) plan.checklist.push(SET_CONSISTENCY_ITEM);
  let shapeRefs: { id: string; image: RgbaImage }[] = [];
  const answer = Array.isArray(planCall.data.references) ? planCall.data.references : [];
  if (candidates.length) refTrace.answer = answer;
  const unmatched: string[] = [];
  for (const pick of answer) {
    const found = matchCandidate(candidates.map((c) => c.candidate), pick?.id);
    const hit = found ? candidates.find((c) => c.candidate.id === found.id) : undefined;
    if (!hit) {
      if (pick?.id !== undefined) unmatched.push(String(pick.id));
      continue;
    }
    const found2 = hit;
    if (shapeRefs.some((s) => s.id === found2.candidate.id) || shapeRefs.length >= maxShape) continue;
    shapeRefs.push({ id: found2.candidate.id, image: found2.image });
    refTrace.shape.push({
      id: found2.candidate.id,
      label: found2.asset.label,
      source: found2.asset.source,
      ...(found2.asset.url ? { url: found2.asset.url } : {}),
      ...(typeof pick.why === "string" && pick.why ? { why: pick.why } : {}),
    });
  }
  if (candidates.length && shapeRefs.length === 0) {
    refTrace.note = unmatched.length
      ? `the planner's reference matched no candidate: ${unmatched.join(", ")}`
      : typeof planCall.data.reference_note === "string" && planCall.data.reference_note
        ? planCall.data.reference_note
        : "the planner chose no reference";
  }
  let keyColor = request.transparent ? plan.key_color ?? Object.keys(backdrops)[0]! : undefined;
  let subjectPrompt = plan.positive_prompt;
  let subjectNegative = plan.negative_prompt;
  const steering = () => referenceSteering(shapeRefs.length, setRefs.length);
  const composePositive = () => {
    const s = steering();
    const subjectText = s ? `${subjectPrompt.replace(/\s+$/, "").replace(/([^.!?])$/, "$1.")} ${s}` : subjectPrompt;
    return composeKeyedPrompt(subjectText, keyColor, promptStyle);
  };
  // Reference images go to the server on the backdrop the asset is rendered
  // on (white for opaque assets); re-sent when the backdrop or the set changes.
  let referenceNames: string[] = [];
  const uploadReferences = async () => {
    const bg = parseHexColor(keyColor ?? "#FFFFFF");
    const all = [...shapeRefs, ...setRefs];
    referenceNames = [];
    for (const [n, r] of all.entries()) {
      const up = await options.comfy.uploadImage(encodePng(compositeOver(r.image, bg)), `comfy-agent-${traceId.slice(0, 8)}-ref${n + 1}-${(keyColor ?? "#FFFFFF").slice(1)}.png`);
      referenceNames.push(up.name);
    }
  };
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
      ...(referenceNames.length ? { references: referenceNames } : {}),
    });

  if (shapeRefs.length || setRefs.length) await uploadReferences();
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
    id: traceId,
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
      ...(cutout ? { cutout } : {}),
    },
    plan: {
      checklist: plan.checklist,
      ...(plan.subject ? { subject: plan.subject } : {}),
      positivePrompt: plan.positive_prompt,
      negativePrompt: plan.negative_prompt,
      ...(keyColor ? { keyColor } : {}),
      ...(plan.notes ? { notes: plan.notes } : {}),
      call: callTrace("planner", planCall),
    },
    ...(wantShape || refOpts.set?.length ? { references: refTrace } : {}),
    rounds: [],
    best: null,
    stopReason: "max-rounds",
    totals: { durationMs: 0, queueWaitMs: 0, inputTokens: 0, outputTokens: 0, reportedCostUsd: null, estimatedCostUsd: 0, costUsd: 0 },
  };
  addUsage(trace, planCall);
  // Progress for hosts that show the loop while it runs: the same trace,
  // rewritten after each round's grading and repair, marked running. The final
  // write at the end replaces it without the mark, also when the loop fails, so
  // a trace is never left running. While running it has no stop reason yet.
  // A failed progress write never stops the loop.
  const writeProgress = async () => {
    await writeFile(path.join(options.workDir, "trace.json"), JSON.stringify({ ...trace, stopReason: null, running: true })).catch(() => undefined);
  };
  const finish = () => {
    for (const r of trace.rounds) r.phase = "done";
    const finished = Date.now();
    trace.finishedAt = new Date(finished).toISOString();
    trace.totals.durationMs = finished - startedAt;
  };
  try {

    let best: { round: number; png: Buffer; graph: ApiWorkflow; score: RoundTrace["score"]; imagePath?: string; copied: boolean } | null = null;
    // The set's icons, shown to the critic for the set_consistency item.
    const setSheet = setRefs.length ? encodePng(contactSheet(setRefs.map((s) => s.image), 256)).toString("base64") : undefined;
    let previousRaw: Buffer | null = null;
    const seenGraphs = new Set<string>();

    for (let round = 1; round <= limits.maxRounds; round += 1) {
      if (options.signal?.aborted) {
        trace.stopReason = "aborted";
        break;
      }
      // Queue waits count only against the wall-clock ceiling, not the active-time budget.
      const wall = Date.now() - startedAt;
      const queued = trace.rounds.reduce((s, r) => s + r.timing.queueWaitMs, 0);
      const active = wall - queued;
      const avgRound = trace.rounds.length ? trace.rounds.reduce((s, r) => s + r.durationMs - r.timing.queueWaitMs, 0) / trace.rounds.length : 0;
      if (round > 1 && wall >= limits.maxWallTimeMs) {
        trace.stopReason = "queue-timeout";
        break;
      }
      if (round > 1 && active + avgRound > limits.maxTimeMs) {
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
      // A matting cut-out runs in the same prompt: the saved image carries the matte as alpha.
      const queuedGraph = cutout?.method === "matte" ? addBackgroundRemoval(doc.graph, cutout.model!) : doc.graph;
      const rt: RoundTrace = {
        round,
        startedAt: new Date(roundStart).toISOString(),
        durationMs: 0,
        timing: { comfyMs: 0, queueWaitMs: 0, postProcessMs: 0, criticMs: 0, plannerMs: 0 },
        graph: structuredClone(queuedGraph),
        settings: describeDoc(doc),
        checks: [],
        score: { passed: 0, total: 0, ratio: 0 },
      };
      for (const prev of trace.rounds) prev.phase = "done";
      rt.phase = "generating";
      rt.queued = true;
      trace.rounds.push(rt);
      log(`round ${round}: generating`);
      await writeProgress();
      let lastProgressWrite = Date.now();

      // 2. Run.
      let rawPng: Buffer;
      let matted: RgbaImage | undefined;
      try {
        assertValidGraph(queuedGraph, info);
        const t = Date.now();
        const run = await options.comfy.run(queuedGraph, {
          validate: false,
          ...(options.runTimeoutMs ? { timeoutMs: options.runTimeoutMs } : {}),
          // The prompt may wait in the queue only until the wall-clock ceiling.
          queueTimeoutMs: Math.max(1, limits.maxWallTimeMs - (Date.now() - startedAt)),
          ...(options.signal ? { signal: options.signal } : {}),
          onProgress: (e) => {
            if (e.type === "started" || e.type === "executing" || e.type === "progress") {
              if (rt.queued) {
                rt.queued = false;
                lastProgressWrite = Date.now();
                void writeProgress(); // leaving the queue is shown at once
              }
            }
            if (e.type === "executing") rt.runningNode = e.nodeId;
            if (e.type === "progress") rt.progress = { ...(e.nodeId ? { nodeId: e.nodeId } : {}), value: e.value, max: e.max };
            // At most one progress write a second; the last step is written by the grading write.
            if ((e.type === "executing" || e.type === "progress") && Date.now() - lastProgressWrite >= 1000) {
              lastProgressWrite = Date.now();
              void writeProgress();
            }
            if (e.type === "started") log(`round ${round}: started after ${Math.round((Date.now() - t) / 1000)} s in the queue`);
            if (e.type === "progress" && e.max > 0 && (e.value === e.max || e.value % 5 === 0)) log(`round ${round}: step ${e.value}/${e.max}`);
          },
        });
        rt.timing.comfyMs = Date.now() - t;
        rt.timing.queueWaitMs = run.queueWaitMs;
        trace.totals.queueWaitMs += run.queueWaitMs;
        const image = run.images.at(-1)!;
        rawPng = image.bytes;
        const raw = decodePng(rawPng);
        if (cutout?.method === "matte") {
          // Keep the cut-out for post-processing; the round's raw image (used for
          // edits and img2img) is the opaque render underneath it.
          matted = raw;
          rawPng = encodePng(opaque(raw));
        }
        rt.generation = { promptId: run.promptId, width: raw.width, height: raw.height, via: run.via };
      } catch (err) {
        rt.error = errorInfo(err);
        rt.durationMs = Date.now() - roundStart;
        const queueTimeout = err instanceof ComfyTimeoutError && err.phase === "queue";
        if (queueTimeout) {
          rt.timing.queueWaitMs = rt.durationMs;
          trace.totals.queueWaitMs += rt.durationMs;
        }
        if (!best) throw err; // nothing to fall back to
        trace.stopReason = options.signal?.aborted ? "aborted" : queueTimeout ? "queue-timeout" : "error";
        break;
      }
      const rawPath = path.join(options.workDir, `round-${round}-raw.png`);
      await writeFile(rawPath, rawPng);
      rt.rawImagePath = recordPath(rawPath);
      previousRaw = rawPng;

      // 3. Post-process in code and run deterministic checks.
      const tp = Date.now();
      const post = postProcess(decodePng(rawPng), request, keyColor, matted ? { matted } : {});
      const finalPng = encodePng(post.image);
      const finalPath = path.join(options.workDir, `round-${round}.png`);
      await writeFile(finalPath, finalPng);
      rt.imagePath = recordPath(finalPath);
      if (post.key) rt.keyColor = post.key;
      if (post.cutout) rt.cutout = post.cutout;
      rt.checks = deterministicChecks(post.image, request, post.key, post.cutout);
      const inUse = [...shapeRefs, ...setRefs];
      if (inUse.length && doc.kind !== "edit") {
        // Near-copy guard: a reference supplies shape, concept or style, never the drawing itself.
        rt.referenceSimilarity = inUse.map((r) => ({ id: r.id, ...referenceSimilarity(post.image, r.image) }));
        const copies = rt.referenceSimilarity.filter((s) => s.nearCopy);
        rt.checks.push({
          id: "reference_copy",
          pass: copies.length === 0,
          detail: copies.length
            ? `the output reproduces ${copies.map((c) => `${c.id} (silhouette ${c.silhouetteIoU.toFixed(2)}, structure ${c.structure.toFixed(2)})`).join(", ")}; it must be a new drawing`
            : `no near-copy of ${inUse.length} reference(s) (limits: silhouette ≥ ${NEAR_COPY_MIN_IOU} and structure ≥ ${NEAR_COPY_MIN_STRUCTURE})`,
        });
      }
      rt.timing.postProcessMs = Date.now() - tp;

      // 4. Critic.
      rt.phase = "grading";
      rt.queued = false;
      delete rt.runningNode;
      await writeProgress();
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
                ...(setSheet ? [{ type: "image" as const, mime: "image/png", base64: setSheet }] : []),
                { type: "text", text: criticUserText(plan.checklist, request, setRefs.length) + reminder },
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

      // 5. Blind recognition: would a viewer who is not told the subject know what it is?
      if (options.recognition !== false && typeof plan.subject === "string" && plan.subject.trim()) {
        const recognised = await recogniseBlind(options.llm, criticModel, criticShape, criticView, plan.subject.trim(), options.criticReasoningEffort, trace);
        if (recognised) rt.checks.push(recognised);
      }

      await writeProgress();
      const passed = rt.checks.filter((c) => c.pass).length + rt.critic.items.filter((c) => c.pass).length;
      const total = rt.checks.length + rt.critic.items.length;
      rt.score = { passed, total, ratio: total ? passed / total : 0 };
      // A near-copy of a reference is never preferred over an original drawing, whatever its score.
      const copied = rt.checks.some((c) => c.id === "reference_copy" && !c.pass);
      if (!best || (best.copied && !copied) || (best.copied === copied && rt.score.ratio > best.score.ratio)) {
        best = { round, png: finalPng, graph: structuredClone(queuedGraph), score: rt.score, imagePath: rt.imagePath, copied };
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
      rt.phase = "repairing";
      await writeProgress();
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
              {
                type: "text",
                text:
                  repairUserText(request, plan.checklist, doc, rt, caps, trace.rounds, keyColor, cutout?.method, backdrops) +
                  ([...shapeRefs, ...setRefs].length
                    ? `\nReferences in use (drop_reference takes these ids): ${[...shapeRefs.map((r) => `${r.id} (shape)`), ...setRefs.map((r) => `${r.id} (set)`)].join(", ")}.`
                    : ""),
              },
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
              if (!backdrops[edit.hex]) {
                throw new GraphEditError(edit.op, `with ${cutout?.method === "matte" ? "matting" : "chroma keying"}, hex must be one of ${Object.keys(backdrops).join(", ")}`);
              }
              keyColor = edit.hex;
              if (referenceNames.length) {
                // The references are sent on the backdrop: send them again on the new one.
                await uploadReferences();
                doc = buildBase(Number(describeDoc(doc).seed ?? seed));
              } else {
                doc = reprompt();
              }
              break;
            case "drop_reference": {
              const before = shapeRefs.length + setRefs.length;
              shapeRefs = shapeRefs.filter((r) => r.id !== edit.id);
              setRefs = setRefs.filter((r) => r.id !== edit.id);
              if (shapeRefs.length + setRefs.length === before) {
                throw new GraphEditError(edit.op, `no reference "${edit.id}" is in use (in use: ${[...shapeRefs, ...setRefs].map((r) => r.id).join(", ") || "none"})`);
              }
              if (setRefs.length === 0) plan.checklist = plan.checklist.filter((c) => c.id !== SET_CONSISTENCY_ITEM.id);
              refTrace.dropped = [...(refTrace.dropped ?? []), { id: edit.id, round }];
              await uploadReferences();
              doc = buildBase(Number(describeDoc(doc).seed ?? seed));
              break;
            }
            case "edit_previous": {
              if (!previousRaw) throw new GraphEditError(edit.op, "no previous image");
              if (!editModel) throw new GraphEditError(edit.op, "no instruction-editing model is installed; use img2img_from_previous or inpaint_previous");
              const up = await options.comfy.uploadImage(previousRaw, `comfy-agent-${trace.id.slice(0, 8)}-r${round}-edit.png`);
              const instruction = keyColor
                ? `${edit.instruction.replace(/\s+$/, "").replace(/([^.!?])$/, "$1.")} Keep the plain flat ${backdropName(keyColor) ?? "key-colour"} background unchanged.`
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
      await writeProgress();
    }

    finish();
    if (!best) throw new ComfyAgentError("COMFY_MISSING_OUTPUT", "the asset loop produced no image");
    trace.best = { round: best.round, score: best.score, ...(best.imagePath ? { imagePath: best.imagePath } : {}) };
    const tracePathAbs = path.join(options.workDir, "trace.json");
    await writeFile(tracePathAbs, JSON.stringify(trace, null, 2));
    return { ok: trace.stopReason === "all-pass", png: best.png, graph: best.graph, trace };
  } catch (err) {
    finish();
    trace.stopReason = options.signal?.aborted ? "aborted" : "error";
    trace.error = errorInfo(err);
    await writeFile(path.join(options.workDir, "trace.json"), JSON.stringify(trace, null, 2)).catch(() => undefined);
    throw err;
  }
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

/**
 * Turn a round's render into the asset: crop/resize to the requested size and,
 * for a transparent asset, cut the subject out. With `matted` (the server's
 * matting output: the render's RGB with the matte as alpha) the backdrop's
 * colour is taken back out of soft edges; otherwise the opaque `raw` render
 * is chroma keyed on `keyColor`.
 */
export function postProcess(
  raw: RgbaImage,
  request: AssetRequest,
  keyColor: string | undefined,
  options: { matted?: RgbaImage } = {},
): { image: RgbaImage; key?: { requested: string; measured: string; borderMatch: number }; cutout?: CutoutInfo } {
  if (request.transparent && keyColor) {
    const nominal = parseHexColor(keyColor);
    const est = estimateKeyColor(raw, nominal);
    const backdrop = { requested: keyColor, measured: toHexColor(est.color), borderMatch: Number(est.borderMatch.toFixed(3)) };
    if (options.matted) {
      return {
        image: coverToSize(decontaminateMatte(options.matted, est.color), request.width, request.height),
        cutout: { method: "matte", backdrop },
      };
    }
    const keyed = chromaKey(raw, est.color, { tolerance: 0.1, softness: 0.1, connectedOnly: true, featherPx: 1, despill: true });
    return {
      image: coverToSize(keyed, request.width, request.height),
      key: backdrop,
      cutout: { method: "chroma-key", backdrop },
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

// A clean matte has no backdrop-coloured ring on its edge, almost no
// backdrop-coloured pixels anywhere, and no translucent area away from the
// edge. Set on 106 real BiRefNet mattes of eval rounds; the 5 that fail were
// each checked by eye (evals/design-assets/MATTE_CALIBRATION.md).
export const MATTE_RESIDUE_MAX_FRINGE = 0.1;
export const MATTE_RESIDUE_MAX_TINTED = 0.02;
export const MATTE_RESIDUE_MAX_HAZE = 0.05;

export function deterministicChecks(
  img: RgbaImage,
  request: AssetRequest,
  key?: { requested: string; measured: string; borderMatch: number },
  cutout?: CutoutInfo,
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
    if (cutout?.method === "matte") {
      const residue = matteResidue(img, parseHexColor(cutout.backdrop.measured));
      checks.push({
        id: "matte_residue",
        pass: residue.fringe <= MATTE_RESIDUE_MAX_FRINGE && residue.backdropTinted <= MATTE_RESIDUE_MAX_TINTED && residue.haze <= MATTE_RESIDUE_MAX_HAZE,
        detail:
          `${(residue.fringe * 100).toFixed(1)}% of edge pixels near the ${cutout.backdrop.requested} backdrop (max ${MATTE_RESIDUE_MAX_FRINGE * 100}%), ` +
          `${(residue.backdropTinted * 100).toFixed(1)}% of visible pixels near it (max ${MATTE_RESIDUE_MAX_TINTED * 100}%), ` +
          `${(residue.haze * 100).toFixed(1)}% partly transparent (max ${MATTE_RESIDUE_MAX_HAZE * 100}%); ` +
          "high values mean a halo, leftover background, shadow or haze around the subject",
      });
    } else if (key) {
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

/**
 * The candidate a planner's pick names. Models answer with the exact id, the
 * number on the sheet (1-based), the id without its `emoji:`/`icon:` prefix,
 * another letter case, or the label; all of these name one candidate.
 */
export function matchCandidate<C extends { id: string; label: string }>(candidates: readonly C[], raw: unknown): C | undefined {
  if (raw === undefined || raw === null) return undefined;
  const s = String(raw).trim().replace(/^#/, "");
  if (/^\d+$/.test(s)) {
    const byNumber = candidates[Number(s) - 1];
    if (byNumber && !candidates.some((c) => c.id.split(":").pop() === s)) return byNumber;
  }
  const k = s.toLowerCase();
  return (
    candidates.find((c) => c.id.toLowerCase() === k) ??
    candidates.find((c) => c.id.split(":").pop()!.toLowerCase() === k) ??
    candidates.find((c) => c.label.toLowerCase() === k)
  );
}

/** Checklist item the loop adds when icons from the same set are used as style references. */
export const SET_CONSISTENCY_ITEM: ChecklistItem = {
  id: "set_consistency",
  text: "The icon matches the style of the icons already made for the same set (rendering, palette, outline weight, lighting, level of detail) while showing its own subject",
};

function chooseCutout(mode: CutoutMode, matteModel: string | undefined): { method: CutoutMethod; model?: string } {
  if (mode === "chroma-key") return { method: "chroma-key" };
  if (matteModel) return { method: "matte", model: matteModel };
  if (mode === "matte") {
    throw new ComfyAgentError("GRAPH_EDIT", "cutout \"matte\" needs a background-removal model in models/background_removal and the core RemoveBackground node");
  }
  return { method: "chroma-key" };
}

/** The same pixels with every pixel fully opaque (the render under a matting cut-out). */
function opaque(img: RgbaImage): RgbaImage {
  const data = new Uint8Array(img.data);
  for (let i = 3; i < data.length; i += 4) data[i] = 255;
  return { width: img.width, height: img.height, data };
}

function normalizePlan(raw: PlanOutput, request: AssetRequest, backdrops: Record<string, string>): PlanOutput {
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
    subject: typeof raw.subject === "string" && raw.subject.trim() ? raw.subject.trim() : request.prompt,
    positive_prompt: typeof raw.positive_prompt === "string" && raw.positive_prompt.trim() ? raw.positive_prompt.trim() : request.prompt,
    negative_prompt: [typeof raw.negative_prompt === "string" ? raw.negative_prompt.trim() : "", request.negative?.trim() ?? ""].filter(Boolean).join(", "),
    ...(key && backdrops[key] ? { key_color: key } : { key_color: undefined }),
  };
}

/**
 * The `recognisable` check: a vision model sees only the image and names it,
 * then a text call decides whether that name is the planned subject. Returns
 * undefined when either call fails, so an outage never fails a round.
 */
async function recogniseBlind(
  llm: AiandClient,
  model: string,
  shape: ApiShape,
  view: Buffer,
  subject: string,
  reasoningEffort: string | undefined,
  trace: AssetTrace,
): Promise<CheckResult | undefined> {
  try {
    const seen = await llm.structured<{ name?: unknown; could_also_be?: unknown }>(
      {
        model,
        system: RECOGNISE_SYSTEM,
        content: [
          { type: "image", mime: "image/png", base64: view.toString("base64") },
          { type: "text", text: `What does this image show? Call ${RECOGNISE_TOOL}.` },
        ],
        schema: RECOGNISE_SCHEMA,
        toolName: RECOGNISE_TOOL,
        maxTokens: 1024,
        ...(reasoningEffort ? { reasoningEffort } : {}),
      },
      shape,
    );
    addUsage(trace, seen);
    const name = typeof seen.data.name === "string" ? seen.data.name.trim() : "";
    if (!name) return undefined;
    const alts = Array.isArray(seen.data.could_also_be) ? seen.data.could_also_be.filter((a): a is string => typeof a === "string").slice(0, 3) : [];
    const verdict = await llm.structured<{ match?: unknown; reason?: unknown }>(
      {
        model,
        system: MATCH_SYSTEM,
        content: [{ type: "text", text: `Intended subject: ${subject}\nThe viewer named it: ${name}${alts.length ? ` (could also be: ${alts.join(", ")})` : ""}\nCall ${MATCH_TOOL}.` }],
        schema: MATCH_SCHEMA,
        toolName: MATCH_TOOL,
        maxTokens: 512,
      },
      shape,
    );
    addUsage(trace, verdict);
    const match = verdict.data.match === true || verdict.data.match === "true";
    return {
      id: "recognisable",
      pass: match,
      detail: match
        ? `a viewer who was not told the subject named it "${name}"`
        : `a viewer who was not told the subject named it "${name}"${alts.length ? ` (or ${alts.join(", ")})` : ""}, not ${subject}`,
    };
  } catch {
    return undefined;
  }
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
  if (r.op === "drop_reference") {
    const id = typeof r.id === "string" ? r.id.trim() : typeof r.text === "string" ? r.text.trim() : "";
    if (!id) throw new GraphEditError("drop_reference", "id must name a reference in use");
    return { op: "drop_reference", id };
  }
  if (r.op === "set_key_color") {
    const hex = String(r.hex ?? "").toUpperCase();
    if (!backdropName(hex)) throw new GraphEditError("set_key_color", `hex must be one of ${[...Object.keys(KEY_COLORS), ...Object.keys(MATTE_BACKDROPS)].join(", ")}`);
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
    ...(r.reportedCost ? { reportedCost: r.reportedCost } : {}),
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
  cutout: CutoutMethod | undefined,
): string {
  const spec = familySpec(family);
  const backdrops = cutout === "matte" ? MATTE_BACKDROPS : KEY_COLORS;
  return [
    `Request: ${request.prompt}`,
    `Output size: ${request.width}x${request.height} px (generated at ${gen.width}x${gen.height}, then cropped/resized in code).`,
    !request.transparent
      ? "Transparent background required: no"
      : cutout === "matte"
        ? "Transparent background required: yes. The subject is rendered on a plain backdrop and cut out with a matting model, which also keeps soft edges, steam and glow. Choose key_color (the backdrop) so it contrasts with the subject."
        : "Transparent background required: yes. The subject is rendered on a flat key colour that is removed by chroma keying in code. Choose key_color far from every subject colour.",
    request.transparent ? `Allowed key_color values: ${Object.entries(backdrops).map(([h, n]) => `${h} (${n})`).join(", ")}.` : "",
    request.style ? `Style and palette notes: ${request.style}` : "",
    request.negative ? `Must not contain: ${request.negative}` : "",
    spec
      ? `Model family: ${family} (${choice?.label ?? spec.label}, distilled, language-model text encoder). Write positive_prompt as plain sentences; steps, cfg, sampler and scheduler are fixed.`
      : `Model family: ${family}. Available samplers: ${caps.samplers.slice(0, 25).join(", ")}. Schedulers: ${caps.schedulers.join(", ")}.`,
    editModel ? `Instruction editing is available for repairs (${editModel.label}).` : "",
    `Call ${PLAN_TOOL}.`,
  ].filter(Boolean).join("\n");
}

function criticUserText(checklist: ChecklistItem[], request: AssetRequest, setCount = 0): string {
  return [
    `Original request: ${request.prompt}`,
    request.transparent
      ? "This is a cut-out asset, shown twice: on white (left) and on near-black (right). Those backgrounds are transparent. " +
        "A visible shadow, glow, tint, haze or coloured fringe around the subject on either side is a defect: fail any item it affects."
      : "",
    setCount
      ? `The second image shows ${setCount} icon(s) already made for the same set. Grade ${SET_CONSISTENCY_ITEM.id} against them; grade every other item on the first image only.`
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
  cutout: CutoutMethod | undefined,
  backdrops: Record<string, string>,
): string {
  const failing = rt.critic?.items.filter((i) => !i.pass) ?? [];
  const history = rounds.slice(0, -1).map((r) => ({
    round: r.round,
    score: `${r.score.passed}/${r.score.total}`,
    applied: r.repair?.applied ?? [],
    rejected: r.repair?.rejected.map((x) => x.error) ?? [],
  }));
  return [
    `Request: ${request.prompt} (${request.width}x${request.height}${
      request.transparent
        ? cutout === "matte"
          ? `, transparent: rendered on a ${keyColor} backdrop and cut out with a matting model`
          : `, transparent via key colour ${keyColor}`
        : ""
    })`,
    request.transparent ? `Allowed set_key_color values: ${Object.keys(backdrops).join(", ")}.` : "",
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
