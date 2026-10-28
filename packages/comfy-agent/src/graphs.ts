// Base graphs are built in code from typed parameters, not copied from one
// frozen JSON export. Each builder returns the API-format graph plus a role
// map (which node is the sampler, the positive prompt, the latent source, ...)
// so typed edits can change the graph without guessing node ids.

import { GraphEditError } from "./errors.js";
import { familySpec, isSplitFamily, splitFamilyOf, type ModelFamily, type SplitFamily, type SplitModelSet } from "./families.js";
import { comboOptions, hasClass, installedModelSets } from "./object-info.js";
import type { ApiWorkflow, ObjectInfo } from "./types.js";
import { isLink } from "./validate.js";

export type { ModelFamily } from "./families.js";
/** `edit`: an instruction-editing graph (the model sees a reference image and an instruction). */
export type GraphKind = "txt2img" | "img2img" | "inpaint" | "edit";

export type Roles = {
  /** CheckpointLoaderSimple for single-file families; UNETLoader for split-loader families. */
  checkpoint?: string;
  /** CLIPLoader (split-loader families). */
  textEncoder?: string;
  /** VAELoader (split-loader families). */
  vae?: string;
  /** Custom-sampler graphs (SamplerCustomAdvanced): RandomNoise, guider, sigmas and sampler-select nodes. */
  noise?: string;
  guider?: string;
  sigmas?: string;
  samplerSelect?: string;
  positive?: string;
  negative?: string;
  /** EmptyLatentImage (txt2img) — owns width/height. */
  latent?: string;
  loadImage?: string;
  /** ImageScale between LoadImage and VAEEncode — owns width/height for img2img/inpaint. */
  inputScale?: string;
  loadMask?: string;
  encode?: string;
  noiseMask?: string;
  sampler?: string;
  decode?: string;
  save?: string;
  loras?: string[];
  /** Second pass added by the upscale edit. */
  upscale?: string[];
};

export type GraphDoc = {
  graph: ApiWorkflow;
  roles: Roles;
  kind: GraphKind;
  family: ModelFamily;
};

export type SamplingParams = {
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  seed: number;
  denoise?: number;
};

export type BaseGraphParams = SamplingParams & {
  /** Checkpoint file (single-file families) or diffusion model file (split-loader families). */
  checkpoint: string;
  positive: string;
  negative: string;
  width: number;
  height: number;
  filenamePrefix?: string;
  /** Split-loader families need these; the family defaults to `detectFamily(checkpoint)`. */
  family?: ModelFamily;
  textEncoder?: string;
  vae?: string;
};

/** The model a base graph runs: a single-file checkpoint or a split-loader model set. */
export type ModelChoice = {
  family: ModelFamily;
  label: string;
  /** Checkpoint or diffusion model file. */
  model: string;
  textEncoder?: string;
  vae?: string;
  /** True when this model can run instruction edits. */
  edit: boolean;
};

// ---------------------------------------------------------------------------
// Families and native sizes

/** SDXL was trained on ~1 MP buckets; generating off-bucket degrades composition. */
export const SDXL_BUCKETS: ReadonlyArray<readonly [number, number]> = [
  [1024, 1024],
  [1152, 896],
  [896, 1152],
  [1216, 832],
  [832, 1216],
  [1344, 768],
  [768, 1344],
  [1536, 640],
  [640, 1536],
];

/** SD 1.x native resolution is 512²; common aspect variants stay near that area. */
export const SD15_BUCKETS: ReadonlyArray<readonly [number, number]> = [
  [512, 512],
  [576, 448],
  [448, 576],
  [640, 448],
  [448, 640],
  [768, 512],
  [512, 768],
  [704, 384],
  [384, 704],
];

export function detectFamily(checkpoint: string): ModelFamily {
  const split = splitFamilyOf(checkpoint);
  if (split) return split;
  const n = checkpoint.toLowerCase();
  if (/(sdxl|sd_xl|\bxl\b|_xl|-xl|xl_|xl-|pony|illustrious|noobai|juggernautxl|realvisxl|animagine)/.test(n)) return "sdxl";
  if (/(v1-5|v1_5|sd15|sd-1|sd_1|1\.5|v1-4|anything-v|dreamshaper_8|realisticvision|deliberate)/.test(n)) return "sd15";
  return "unknown";
}

/**
 * SD 1.5 uses its 512² buckets. SDXL and the split-loader families (Z-Image,
 * FLUX.2 klein, Ming-Image) are all trained around 1 MP; their example
 * workflows default to 1024², so they share the SDXL aspect buckets.
 */
export function bucketsFor(family: ModelFamily): ReadonlyArray<readonly [number, number]> {
  return family === "sd15" ? SD15_BUCKETS : SDXL_BUCKETS;
}

/**
 * Generation size for a requested output size: the native bucket whose aspect
 * ratio is closest to the request, scaled down toward the requested area when
 * the request is small (never below what the family handles well), rounded to
 * multiples of 64. The asset is cropped/resized to the exact request after.
 */
export function planGenerationSize(family: ModelFamily, width: number, height: number): { width: number; height: number; bucket: readonly [number, number] } {
  const target = Math.log(width / height);
  let best = bucketsFor(family)[0]!;
  for (const b of bucketsFor(family)) {
    if (Math.abs(Math.log(b[0] / b[1]) - target) < Math.abs(Math.log(best[0] / best[1]) - target)) best = b;
  }
  const minScale = family === "sd15" ? 1 : 0.75;
  const scale = Math.min(1, Math.max(minScale, Math.sqrt((width * height) / (best[0] * best[1]))));
  const round64 = (v: number) => Math.max(64, Math.round(v / 64) * 64);
  return { width: round64(best[0] * scale), height: round64(best[1] * scale), bucket: best };
}

export function defaultSampling(family: ModelFamily, info?: ObjectInfo): SamplingParams {
  const spec = familySpec(family);
  if (spec) return { ...spec.sampling, seed: 0 };
  const samplers = info ? comboOptions(info, "KSampler", "sampler_name") : [];
  const schedulers = info ? comboOptions(info, "KSampler", "scheduler") : [];
  const pick = (want: string[], have: string[]) => want.find((w) => have.length === 0 || have.includes(w)) ?? have[0] ?? want[0]!;
  return {
    steps: family === "sd15" ? 25 : 24,
    cfg: family === "sd15" ? 7 : 6.5,
    sampler: pick(["dpmpp_2m", "euler"], samplers),
    scheduler: pick(["karras", "normal"], schedulers),
    seed: 0,
  };
}

// ---------------------------------------------------------------------------
// Builders

function samplerNode(p: SamplingParams, model: [string, number], pos: string, neg: string, latent: [string, number]) {
  return {
    class_type: "KSampler",
    inputs: {
      model,
      positive: [pos, 0],
      negative: [neg, 0],
      latent_image: latent,
      seed: p.seed,
      steps: p.steps,
      cfg: p.cfg,
      sampler_name: p.sampler,
      scheduler: p.scheduler,
      denoise: p.denoise ?? 1,
    },
  };
}

function common(p: BaseGraphParams): ApiWorkflow {
  return {
    ckpt: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: p.checkpoint }, _meta: { title: "Checkpoint" } },
    pos: { class_type: "CLIPTextEncode", inputs: { text: p.positive, clip: ["ckpt", 1] }, _meta: { title: "Positive" } },
    neg: { class_type: "CLIPTextEncode", inputs: { text: p.negative, clip: ["ckpt", 1] }, _meta: { title: "Negative" } },
    decode: { class_type: "VAEDecode", inputs: { samples: ["sampler", 0], vae: ["ckpt", 2] } },
    save: { class_type: "SaveImage", inputs: { images: ["decode", 0], filename_prefix: p.filenamePrefix ?? "comfy-agent" } },
  };
}

const COMMON_ROLES: Roles = { checkpoint: "ckpt", positive: "pos", negative: "neg", sampler: "sampler", decode: "decode", save: "save" };

export function buildTxt2Img(p: BaseGraphParams): GraphDoc {
  const family = p.family ?? detectFamily(p.checkpoint);
  if (isSplitFamily(family)) return buildSplitTxt2Img(family, p);
  const graph = common(p);
  graph.latent = { class_type: "EmptyLatentImage", inputs: { width: p.width, height: p.height, batch_size: 1 } };
  graph.sampler = samplerNode({ ...p, denoise: 1 }, ["ckpt", 0], "pos", "neg", ["latent", 0]);
  return { graph, roles: { ...COMMON_ROLES, latent: "latent" }, kind: "txt2img", family: detectFamily(p.checkpoint) };
}

/** img2img from an image already on the server (`LoadImage` name), scaled to the generation size. */
export function buildImg2Img(p: BaseGraphParams & { image: string; denoise: number }): GraphDoc {
  const graph = common(p);
  graph.load = { class_type: "LoadImage", inputs: { image: p.image } };
  graph.scale = { class_type: "ImageScale", inputs: { image: ["load", 0], upscale_method: "lanczos", width: p.width, height: p.height, crop: "center" } };
  graph.encode = { class_type: "VAEEncode", inputs: { pixels: ["scale", 0], vae: ["ckpt", 2] } };
  graph.sampler = samplerNode(p, ["ckpt", 0], "pos", "neg", ["encode", 0]);
  return {
    graph,
    roles: { ...COMMON_ROLES, loadImage: "load", inputScale: "scale", encode: "encode" },
    kind: "img2img",
    family: detectFamily(p.checkpoint),
  };
}

/**
 * Inpaint with a mask image on the server. Uses `SetLatentNoiseMask`, which
 * works with ordinary (non-inpainting) checkpoints at denoise < 1.
 */
export function buildInpaint(p: BaseGraphParams & { image: string; mask: string; denoise: number }): GraphDoc {
  const doc = buildImg2Img(p);
  const g = doc.graph;
  g.mask = { class_type: "LoadImageMask", inputs: { image: p.mask, channel: "red" } };
  g.maskScale = {
    class_type: "ImageScale",
    inputs: { image: ["maskImage", 0], upscale_method: "bilinear", width: p.width, height: p.height, crop: "center" },
  };
  // LoadImageMask returns MASK; scale it via MASK→IMAGE→MASK so it matches the latent.
  g.maskImage = { class_type: "MaskToImage", inputs: { mask: ["mask", 0] } };
  g.maskBack = { class_type: "ImageToMask", inputs: { image: ["maskScale", 0], channel: "red" } };
  g.noiseMask = { class_type: "SetLatentNoiseMask", inputs: { samples: ["encode", 0], mask: ["maskBack", 0] } };
  (g.sampler!.inputs as Record<string, unknown>).latent_image = ["noiseMask", 0];
  return { ...doc, roles: { ...doc.roles, loadMask: "mask", noiseMask: "noiseMask" }, kind: "inpaint" };
}

// ---------------------------------------------------------------------------
// Split-loader families (diffusion model + text encoder + VAE as separate files)

function splitLoaders(family: SplitFamily, model: string, textEncoder: string | undefined, vae: string | undefined): ApiWorkflow {
  const spec = familySpec(family)!;
  if (!textEncoder || !vae) {
    throw new GraphEditError("build_graph", `${spec.label} needs a text encoder and a VAE file as well as the diffusion model`);
  }
  return {
    unet: { class_type: "UNETLoader", inputs: { unet_name: model, weight_dtype: "default" }, _meta: { title: "Diffusion model" } },
    clip: { class_type: "CLIPLoader", inputs: { clip_name: textEncoder, type: spec.clipType }, _meta: { title: "Text encoder" } },
    vae: { class_type: "VAELoader", inputs: { vae_name: vae } },
  };
}

const SPLIT_ROLES: Roles = { checkpoint: "unet", textEncoder: "clip", vae: "vae", positive: "pos", sampler: "sampler", decode: "decode", save: "save" };
const CUSTOM_ROLES: Roles = { ...SPLIT_ROLES, noise: "noise", guider: "guider", sigmas: "sigmas", samplerSelect: "select" };

function splitTail(g: ApiWorkflow, prefix: string | undefined): void {
  g.decode = { class_type: "VAEDecode", inputs: { samples: ["sampler", 0], vae: ["vae", 0] } };
  g.save = { class_type: "SaveImage", inputs: { images: ["decode", 0], filename_prefix: prefix ?? "comfy-agent" } };
}

function customSampler(g: ApiWorkflow, latent: [string, number]): void {
  g.sampler = {
    class_type: "SamplerCustomAdvanced",
    inputs: { noise: ["noise", 0], guider: ["guider", 0], sampler: ["select", 0], sigmas: ["sigmas", 0], latent_image: latent },
  };
}

/**
 * Text-to-image for a split-loader family, following its official example
 * workflow (see `FAMILY_SPECS[].source`):
 * - Z-Image-Turbo: ModelSamplingAuraFlow shift 3, ConditioningZeroOut negative, EmptySD3LatentImage, KSampler.
 * - FLUX.2 [klein] 4B: EmptyFlux2LatentImage, Flux2Scheduler, CFGGuider, SamplerCustomAdvanced.
 * - Ming-Image-0.1-Design: ModelSamplingFlux(1.15, 0.5, w, h), BasicScheduler, BasicGuider, SamplerCustomAdvanced.
 * The optional prompt-enhancer LLMs in those templates are left out: the
 * planner writes the prompt.
 */
function buildSplitTxt2Img(family: SplitFamily, p: BaseGraphParams): GraphDoc {
  const g = splitLoaders(family, p.checkpoint, p.textEncoder, p.vae);
  g.pos = { class_type: "CLIPTextEncode", inputs: { text: p.positive, clip: ["clip", 0] }, _meta: { title: "Positive" } };
  let roles: Roles;
  if (family === "zimage") {
    g.shift = { class_type: "ModelSamplingAuraFlow", inputs: { model: ["unet", 0], shift: 3 } };
    g.neg = { class_type: "ConditioningZeroOut", inputs: { conditioning: ["pos", 0] } };
    g.latent = { class_type: "EmptySD3LatentImage", inputs: { width: p.width, height: p.height, batch_size: 1 } };
    g.sampler = samplerNode({ ...p, denoise: 1 }, ["shift", 0], "pos", "neg", ["latent", 0]);
    roles = { ...SPLIT_ROLES, latent: "latent" };
  } else if (family === "flux2_klein") {
    g.neg = { class_type: "ConditioningZeroOut", inputs: { conditioning: ["pos", 0] } };
    g.latent = { class_type: "EmptyFlux2LatentImage", inputs: { width: p.width, height: p.height, batch_size: 1 } };
    g.sigmas = { class_type: "Flux2Scheduler", inputs: { steps: p.steps, width: p.width, height: p.height } };
    g.noise = { class_type: "RandomNoise", inputs: { noise_seed: p.seed } };
    g.select = { class_type: "KSamplerSelect", inputs: { sampler_name: p.sampler } };
    g.guider = { class_type: "CFGGuider", inputs: { model: ["unet", 0], positive: ["pos", 0], negative: ["neg", 0], cfg: p.cfg } };
    customSampler(g, ["latent", 0]);
    roles = { ...CUSTOM_ROLES, latent: "latent" };
  } else {
    g.shift = { class_type: "ModelSamplingFlux", inputs: { model: ["unet", 0], max_shift: 1.15, base_shift: 0.5, width: p.width, height: p.height } };
    g.latent = { class_type: "EmptyLatentImage", inputs: { width: p.width, height: p.height, batch_size: 1 } };
    g.sigmas = { class_type: "BasicScheduler", inputs: { model: ["shift", 0], scheduler: p.scheduler, steps: p.steps, denoise: 1 } };
    g.noise = { class_type: "RandomNoise", inputs: { noise_seed: p.seed } };
    g.select = { class_type: "KSamplerSelect", inputs: { sampler_name: p.sampler } };
    g.guider = { class_type: "BasicGuider", inputs: { model: ["shift", 0], conditioning: ["pos", 0] } };
    customSampler(g, ["latent", 0]);
    roles = { ...CUSTOM_ROLES, latent: "latent" };
  }
  splitTail(g, p.filenamePrefix);
  return { graph: g, roles, kind: "txt2img", family };
}

export type EditGraphParams = {
  family: SplitFamily;
  model: string;
  textEncoder: string;
  vae: string;
  /** Image already on the server (`LoadImage` name). */
  image: string;
  /** Plain-language edit instruction ("make the background plain white"). */
  instruction: string;
  seed: number;
  steps?: number;
  /** Reference is scaled to about this many megapixels; the output follows its size. */
  megapixels?: number;
  filenamePrefix?: string;
};

/**
 * Instruction edit of an uploaded image with an editing-capable family:
 * - FLUX.2 [klein] 4B (image_flux2_klein_image_edit_4b_distilled.json): the reference is scaled to
 *   about 1 MP, VAE-encoded and attached to both conditionings with ReferenceLatent; the latent and
 *   Flux2Scheduler take the reference size; 4 steps, CFGGuider 1, euler.
 * - Ming-Image-0.1-Design (image_ming_image_01_design_image_edit.json): TextEncodeMingImageEdit sees the
 *   reference; the latent matches its size; BasicScheduler 12 steps, SamplerLCM(1.02, 1, 0), BasicGuider.
 * The reference is scaled in multiples of 16 so the latent size matches it exactly.
 */
export function buildEdit(p: EditGraphParams): GraphDoc {
  const spec = familySpec(p.family);
  if (!spec?.editNodes) throw new GraphEditError("edit_with_instruction", `${spec?.label ?? p.family} cannot edit images`);
  const g = splitLoaders(p.family, p.model, p.textEncoder, p.vae);
  g.load = { class_type: "LoadImage", inputs: { image: p.image } };
  g.refScale = {
    class_type: "ImageScaleToTotalPixels",
    inputs: { image: ["load", 0], upscale_method: "lanczos", megapixels: p.megapixels ?? 1, resolution_steps: 16 },
  };
  g.refSize = { class_type: "GetImageSize", inputs: { image: ["refScale", 0] } };
  g.noise = { class_type: "RandomNoise", inputs: { noise_seed: p.seed } };
  const steps = p.steps ?? spec.sampling.steps;
  if (p.family === "flux2_klein") {
    g.pos = { class_type: "CLIPTextEncode", inputs: { text: p.instruction, clip: ["clip", 0] }, _meta: { title: "Instruction" } };
    g.neg = { class_type: "ConditioningZeroOut", inputs: { conditioning: ["pos", 0] } };
    g.refEncode = { class_type: "VAEEncode", inputs: { pixels: ["refScale", 0], vae: ["vae", 0] } };
    g.refPos = { class_type: "ReferenceLatent", inputs: { conditioning: ["pos", 0], latent: ["refEncode", 0] } };
    g.refNeg = { class_type: "ReferenceLatent", inputs: { conditioning: ["neg", 0], latent: ["refEncode", 0] } };
    g.latent = { class_type: "EmptyFlux2LatentImage", inputs: { width: ["refSize", 0], height: ["refSize", 1], batch_size: 1 } };
    g.sigmas = { class_type: "Flux2Scheduler", inputs: { steps, width: ["refSize", 0], height: ["refSize", 1] } };
    g.select = { class_type: "KSamplerSelect", inputs: { sampler_name: spec.sampling.sampler } };
    g.guider = { class_type: "CFGGuider", inputs: { model: ["unet", 0], positive: ["refPos", 0], negative: ["refNeg", 0], cfg: spec.sampling.cfg } };
  } else {
    g.shift = { class_type: "ModelSamplingFlux", inputs: { model: ["unet", 0], max_shift: 1.15, base_shift: 0.5, width: ["refSize", 0], height: ["refSize", 1] } };
    g.pos = {
      class_type: "TextEncodeMingImageEdit",
      inputs: { clip: ["clip", 0], vae: ["vae", 0], prompt: p.instruction, "images.image_1": ["refScale", 0] },
      _meta: { title: "Instruction" },
    };
    g.latent = { class_type: "EmptyLatentImage", inputs: { width: ["refSize", 0], height: ["refSize", 1], batch_size: 1 } };
    g.sigmas = { class_type: "BasicScheduler", inputs: { model: ["shift", 0], scheduler: spec.sampling.scheduler, steps, denoise: 1 } };
    g.select = { class_type: "SamplerLCM", inputs: { s_noise: 1.02, s_noise_end: 1, noise_clip_std: 0 } };
    g.guider = { class_type: "BasicGuider", inputs: { model: ["shift", 0], conditioning: ["pos", 0] } };
  }
  customSampler(g, ["latent", 0]);
  splitTail(g, p.filenamePrefix);
  return { graph: g, roles: { ...CUSTOM_ROLES, loadImage: "load" }, kind: "edit", family: p.family };
}

// ---------------------------------------------------------------------------
// Model choice

/**
 * Every model this server can run a base graph with, strongest first: complete
 * split-loader sets in `FAMILY_SPECS` order, then single-file checkpoints
 * (SDXL before others). The first entry is the default.
 */
export function installedModels(info: ObjectInfo): ModelChoice[] {
  const split: ModelChoice[] = installedModelSets(info).map((m: SplitModelSet) => ({
    family: m.family,
    label: m.label,
    model: m.model,
    textEncoder: m.textEncoder,
    vae: m.vae,
    edit: m.edit,
  }));
  const core = ["CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"];
  const ckpts = core.every((c) => hasClass(info, c)) ? comboOptions(info, "CheckpointLoaderSimple", "ckpt_name") : [];
  const ordered = [...ckpts.filter((c) => detectFamily(c) === "sdxl"), ...ckpts.filter((c) => detectFamily(c) !== "sdxl")];
  return [...split, ...ordered.map((c) => ({ family: detectFamily(c), label: c, model: c, edit: false }))];
}

/**
 * The model to generate with: `preferred` (a checkpoint or diffusion model
 * file name) when it is installed as a complete set, else the strongest
 * installed model, with SDXL as the fallback.
 */
export function chooseModel(info: ObjectInfo, preferred?: string): ModelChoice {
  const all = installedModels(info);
  if (preferred) {
    const found = all.find((m) => m.model === preferred);
    if (found) return found;
    throw new GraphEditError(
      "choose_model",
      `model "${preferred}" is not installed as a complete set (diffusion model, text encoder and VAE); available: ${all.map((m) => m.model).join(", ") || "none"}`,
    );
  }
  const first = all[0];
  if (!first) throw new GraphEditError("choose_model", "the server has no usable models: no checkpoint in models/checkpoints and no complete diffusion model set");
  return first;
}

/** The strongest installed model that can run instruction edits, if any. */
export function chooseEditModel(info: ObjectInfo): ModelChoice | undefined {
  return installedModels(info).find((m) => m.edit);
}

/** Which base graphs this server can run. */
export function availableBaseGraphs(info: ObjectInfo): GraphKind[] {
  const out: GraphKind[] = [];
  const core = ["CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "VAEDecode", "SaveImage"];
  if (core.every((c) => hasClass(info, c)) && hasClass(info, "EmptyLatentImage")) out.push("txt2img");
  if (core.every((c) => hasClass(info, c)) && ["LoadImage", "ImageScale", "VAEEncode"].every((c) => hasClass(info, c))) out.push("img2img");
  if (out.includes("img2img") && ["LoadImageMask", "MaskToImage", "ImageToMask", "SetLatentNoiseMask"].every((c) => hasClass(info, c))) out.push("inpaint");
  const sets = installedModelSets(info);
  if (sets.length > 0 && !out.includes("txt2img")) out.unshift("txt2img");
  if (sets.some((m) => m.edit)) out.push("edit");
  return out;
}

/** Pick the checkpoint: the preferred one if installed, else an SDXL one, else the first. */
export function chooseCheckpoint(info: ObjectInfo, preferred?: string): string {
  const all = comboOptions(info, "CheckpointLoaderSimple", "ckpt_name");
  if (all.length === 0) throw new GraphEditError("choose_checkpoint", "the server has no checkpoints in models/checkpoints");
  if (preferred) {
    if (all.includes(preferred)) return preferred;
    throw new GraphEditError("choose_checkpoint", `checkpoint "${preferred}" is not installed; available: ${all.join(", ")}`);
  }
  return all.find((c) => detectFamily(c) === "sdxl") ?? all[0]!;
}

// ---------------------------------------------------------------------------
// Role inference for user-supplied graphs

const SAMPLER_CLASSES = ["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"];

function upstream(graph: ApiWorkflow, nodeId: string, input: string): string | undefined {
  const v = graph[nodeId]?.inputs[input];
  return isLink(v) ? String(v[0]) : undefined;
}

/** Follow a chain upstream until `match` accepts a node (e.g. a text encoder behind ControlNet nodes). */
function walkUp(graph: ApiWorkflow, start: string | undefined, inputs: string[], match: (id: string) => boolean): string | undefined {
  const seen = new Set<string>();
  let cur = start;
  while (cur && !seen.has(cur)) {
    seen.add(cur);
    if (match(cur)) return cur;
    const node = graph[cur];
    if (!node) return undefined;
    cur = inputs.map((name) => upstream(graph, cur!, name)).find(Boolean);
  }
  return undefined;
}

export function inferRoles(graph: ApiWorkflow): Roles {
  const roles: Roles = {};
  const ids = Object.keys(graph);
  const saves = ids.filter((id) => graph[id]!.class_type === "SaveImage");
  roles.save = saves[0] ?? ids.find((id) => graph[id]!.class_type === "PreviewImage");
  // The sampler feeding the saved image, else the first sampler.
  const decodeForSave = roles.save ? upstream(graph, roles.save, "images") : undefined;
  if (decodeForSave && graph[decodeForSave]?.class_type === "VAEDecode") roles.decode = decodeForSave;
  const fromDecode = roles.decode ? upstream(graph, roles.decode, "samples") : undefined;
  roles.sampler = walkUp(graph, fromDecode, ["samples", "latent_image", "latent"], (id) => SAMPLER_CLASSES.includes(graph[id]!.class_type))
    ?? ids.find((id) => SAMPLER_CLASSES.includes(graph[id]!.class_type));
  if (!roles.sampler) return roles;
  // SamplerCustomAdvanced: noise, guider, sigmas and sampler arrive as separate nodes,
  // and the conditionings and model hang off the guider.
  const custom = graph[roles.sampler]!.class_type === "SamplerCustomAdvanced";
  const condSource = custom ? upstream(graph, roles.sampler, "guider") : roles.sampler;
  if (custom) {
    roles.noise = upstream(graph, roles.sampler, "noise");
    roles.guider = condSource;
    roles.sigmas = upstream(graph, roles.sampler, "sigmas");
    roles.samplerSelect = upstream(graph, roles.sampler, "sampler");
  }
  const textEncoder = (id: string) =>
    typeof graph[id]!.inputs.text === "string" || typeof graph[id]!.inputs.text_g === "string" || (typeof graph[id]!.inputs.prompt === "string" && isLink(graph[id]!.inputs.clip));
  // ConditioningZeroOut means "no negative prompt": stop there rather than walking on to the positive encoder.
  const zeroed = (id: string) => graph[id]!.class_type === "ConditioningZeroOut";
  const walkCond = (start: string | undefined, inputs: string[]) => {
    const found = walkUp(graph, start, inputs, (id) => zeroed(id) || textEncoder(id));
    return found && !zeroed(found) ? found : undefined;
  };
  if (condSource) {
    roles.positive = walkCond(upstream(graph, condSource, "positive") ?? upstream(graph, condSource, "conditioning"), ["conditioning", "conditioning_1", "positive"]);
    roles.negative = walkCond(upstream(graph, condSource, "negative"), ["conditioning", "conditioning_1", "negative"]);
  }
  const latentId = walkUp(
    graph,
    upstream(graph, roles.sampler, "latent_image"),
    ["samples"],
    (id) => typeof graph[id]!.inputs.width === "number" && typeof graph[id]!.inputs.height === "number" && graph[id]!.class_type.includes("Latent"),
  );
  if (latentId) roles.latent = latentId;
  const modelFrom = condSource && custom ? condSource : roles.sampler;
  roles.checkpoint = walkUp(graph, upstream(graph, modelFrom, "model"), ["model"], (id) => typeof graph[id]!.inputs.ckpt_name === "string" || typeof graph[id]!.inputs.unet_name === "string");
  if (roles.positive) {
    const clip = upstream(graph, roles.positive, "clip");
    if (clip && graph[clip]?.class_type === "CLIPLoader") roles.textEncoder = clip;
  }
  if (roles.decode) {
    const vae = upstream(graph, roles.decode, "vae");
    if (vae && graph[vae]?.class_type === "VAELoader") roles.vae = vae;
  }
  return roles;
}

/** Checkpoint or diffusion model file the doc's loader names. */
export function modelFileOf(doc: Pick<GraphDoc, "graph" | "roles">): string {
  const node = doc.roles.checkpoint ? doc.graph[doc.roles.checkpoint] : undefined;
  const name = node?.inputs.ckpt_name ?? node?.inputs.unet_name;
  return typeof name === "string" ? name : "";
}

const EDIT_CLASSES = new Set(["ReferenceLatent", "TextEncodeMingImageEdit", "TextEncodeQwenImageEdit", "TextEncodeQwenImageEditPlus"]);

export function docFromGraph(graph: ApiWorkflow): GraphDoc {
  const roles = inferRoles(graph);
  const model = modelFileOf({ graph, roles });
  const hasMask = Object.values(graph).some((n) => n.class_type === "SetLatentNoiseMask" || n.class_type === "VAEEncodeForInpaint");
  const hasLoad = Object.values(graph).some((n) => n.class_type === "LoadImage");
  const isEdit = Object.values(graph).some((n) => EDIT_CLASSES.has(n.class_type));
  if (isEdit) {
    const load = Object.entries(graph).find(([, n]) => n.class_type === "LoadImage")?.[0];
    if (load) roles.loadImage = load;
  }
  const kind: GraphKind = isEdit ? "edit" : hasMask ? "inpaint" : hasLoad && !roles.latent ? "img2img" : "txt2img";
  return { graph, roles, kind, family: detectFamily(model) };
}
