// Base graphs are built in code from typed parameters, not copied from one
// frozen JSON export. Each builder returns the API-format graph plus a role
// map (which node is the sampler, the positive prompt, the latent source, ...)
// so typed edits can change the graph without guessing node ids.

import { GraphEditError } from "./errors.js";
import { comboOptions, hasClass } from "./object-info.js";
import type { ApiWorkflow, ObjectInfo } from "./types.js";
import { isLink } from "./validate.js";

export type ModelFamily = "sd15" | "sdxl" | "unknown";
export type GraphKind = "txt2img" | "img2img" | "inpaint";

export type Roles = {
  checkpoint?: string;
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
  checkpoint: string;
  positive: string;
  negative: string;
  width: number;
  height: number;
  filenamePrefix?: string;
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
  const n = checkpoint.toLowerCase();
  if (/(sdxl|sd_xl|\bxl\b|_xl|-xl|xl_|xl-|pony|illustrious|noobai|juggernautxl|realvisxl|animagine)/.test(n)) return "sdxl";
  if (/(v1-5|v1_5|sd15|sd-1|sd_1|1\.5|v1-4|anything-v|dreamshaper_8|realisticvision|deliberate)/.test(n)) return "sd15";
  return "unknown";
}

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

/** Which base graphs this server can run. */
export function availableBaseGraphs(info: ObjectInfo): GraphKind[] {
  const out: GraphKind[] = [];
  const core = ["CheckpointLoaderSimple", "CLIPTextEncode", "KSampler", "VAEDecode", "SaveImage"];
  if (core.every((c) => hasClass(info, c)) && hasClass(info, "EmptyLatentImage")) out.push("txt2img");
  if (core.every((c) => hasClass(info, c)) && ["LoadImage", "ImageScale", "VAEEncode"].every((c) => hasClass(info, c))) out.push("img2img");
  if (out.includes("img2img") && ["LoadImageMask", "MaskToImage", "ImageToMask", "SetLatentNoiseMask"].every((c) => hasClass(info, c))) out.push("inpaint");
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
  const textEncoder = (id: string) => typeof graph[id]!.inputs.text === "string" || typeof graph[id]!.inputs.text_g === "string";
  roles.positive = walkUp(graph, upstream(graph, roles.sampler, "positive"), ["conditioning", "conditioning_1", "positive"], textEncoder);
  roles.negative = walkUp(graph, upstream(graph, roles.sampler, "negative"), ["conditioning", "conditioning_1", "negative"], textEncoder);
  const latentId = walkUp(
    graph,
    upstream(graph, roles.sampler, "latent_image"),
    ["samples"],
    (id) => typeof graph[id]!.inputs.width === "number" && typeof graph[id]!.inputs.height === "number" && graph[id]!.class_type.includes("Latent"),
  );
  if (latentId) roles.latent = latentId;
  roles.checkpoint = walkUp(graph, upstream(graph, roles.sampler, "model"), ["model"], (id) => typeof graph[id]!.inputs.ckpt_name === "string");
  return roles;
}

export function docFromGraph(graph: ApiWorkflow): GraphDoc {
  const roles = inferRoles(graph);
  const ckpt = roles.checkpoint ? String(graph[roles.checkpoint]!.inputs.ckpt_name) : "";
  const hasMask = Object.values(graph).some((n) => n.class_type === "SetLatentNoiseMask" || n.class_type === "VAEEncodeForInpaint");
  const hasLoad = Object.values(graph).some((n) => n.class_type === "LoadImage");
  return { graph, roles, kind: hasMask ? "inpaint" : hasLoad && !roles.latent ? "img2img" : "txt2img", family: detectFamily(ckpt) };
}
