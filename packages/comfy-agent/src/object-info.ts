import { installedSplitModels, type SplitModelSet } from "./families.js";
import type { ObjectInfo, ObjectInfoInputSpec, ObjectInfoNode } from "./types.js";

// Input types that ComfyUI renders as widgets (values stored inline) rather
// than as sockets (values arrive over a link).
const PRIMITIVE_TYPES = new Set(["INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"]);

export type InputSpec = {
  name: string;
  required: boolean;
  /** Socket type ("MODEL", "LATENT", ...) or primitive type ("INT", "COMBO", ...). */
  type: string;
  /** Allowed values for COMBO inputs. */
  options?: string[];
  default?: unknown;
  min?: number;
  max?: number;
  /** `LoadImage.image` and friends: the enum lists files in input/, and uploads extend it. */
  imageUpload: boolean;
  /** INT inputs with a "control after generate" companion widget in UI-format JSON. */
  controlAfterGenerate: boolean;
  /** True when the value is a widget (inline), false for a socket. */
  isWidget: boolean;
  /** `forceInput` primitives are sockets in the UI even though their type is primitive. */
  forceInput: boolean;
};

export function parseInputSpec(name: string, raw: ObjectInfoInputSpec, required: boolean): InputSpec {
  const head = raw[0];
  const opts = (raw[1] ?? {}) as Record<string, unknown>;
  let type: string;
  let options: string[] | undefined;
  if (Array.isArray(head)) {
    // Legacy COMBO shape: [["a", "b"], {...}]
    type = "COMBO";
    options = head.map((v) => String(v));
  } else if (head === "COMBO") {
    type = "COMBO";
    const listed = opts.options;
    options = Array.isArray(listed) ? listed.map((v) => String(v)) : [];
  } else {
    type = String(head);
  }
  const forceInput = opts.forceInput === true;
  const spec: InputSpec = {
    name,
    required,
    type,
    imageUpload: opts.image_upload === true,
    controlAfterGenerate: opts.control_after_generate === true || ((name === "seed" || name === "noise_seed") && type === "INT"),
    isWidget: PRIMITIVE_TYPES.has(type) && !forceInput,
    forceInput,
  };
  if (options) spec.options = options;
  if ("default" in opts) spec.default = opts.default;
  if (typeof opts.min === "number") spec.min = opts.min;
  if (typeof opts.max === "number") spec.max = opts.max;
  return spec;
}

/** All declared inputs for a node class, required first, in declaration order. */
export function inputSpecs(node: ObjectInfoNode): InputSpec[] {
  const out: InputSpec[] = [];
  for (const bucket of ["required", "optional"] as const) {
    const group = node.input?.[bucket] ?? {};
    const order = node.input_order?.[bucket] ?? Object.keys(group);
    for (const name of order) {
      const raw = group[name];
      if (raw) out.push(parseInputSpec(name, raw, bucket === "required"));
    }
  }
  return out;
}

export function inputSpec(info: ObjectInfo, classType: string, input: string): InputSpec | undefined {
  const node = info[classType];
  if (!node) return undefined;
  return inputSpecs(node).find((s) => s.name === input);
}

/** Widget inputs in the order a UI-format node stores them in `widgets_values`. */
export function widgetOrder(node: ObjectInfoNode): InputSpec[] {
  return inputSpecs(node).filter((s) => s.isWidget);
}

export function comboOptions(info: ObjectInfo, classType: string, input: string): string[] {
  return inputSpec(info, classType, input)?.options ?? [];
}

export function hasClass(info: ObjectInfo, classType: string): boolean {
  return Object.prototype.hasOwnProperty.call(info, classType);
}

/**
 * Complete split-loader model sets (diffusion model + text encoder + VAE,
 * plus the node classes their graphs need) installed on this server, in
 * preference order.
 */
export function installedModelSets(info: ObjectInfo): SplitModelSet[] {
  return installedSplitModels(
    {
      diffusionModels: comboOptions(info, "UNETLoader", "unet_name"),
      textEncoders: comboOptions(info, "CLIPLoader", "clip_name"),
      vaes: comboOptions(info, "VAELoader", "vae_name"),
    },
    (classes) => classes.every((c) => hasClass(info, c)),
  );
}

/** Summary used by `comfy_inspect` and by the planner prompt. */
export type ServerCapabilities = {
  checkpoints: string[];
  /** Files in models/diffusion_models (split-loader families load these with UNETLoader). */
  diffusionModels: string[];
  /** Complete split-loader model sets, strongest first. */
  modelSets: SplitModelSet[];
  /** What a base graph uses when no model is named: the first model set, else an SDXL checkpoint. */
  defaultModel: { family: string; model: string } | null;
  /** The model that runs instruction edits, when one is installed. */
  editModel: { family: string; model: string } | null;
  /** Matting model for transparent assets (core RemoveBackground node), when one is installed. */
  backgroundRemoval: { model: string } | null;
  loras: string[];
  upscaleModels: string[];
  samplers: string[];
  schedulers: string[];
  nodeCount: number;
  supports: {
    txt2img: boolean;
    img2img: boolean;
    inpaint: boolean;
    /** Instruction-based editing ("make the background white") with an edit-capable model. */
    edit: boolean;
    lora: boolean;
    latentUpscale: boolean;
    modelUpscale: boolean;
  };
};

/** Core ComfyUI nodes a matting cut-out needs (no custom node). */
export const MATTING_CLASSES = ["LoadBackgroundRemovalModel", "RemoveBackground", "InvertMask", "JoinImageWithAlpha"] as const;

/**
 * The installed background-removal (matting) model, when the server has one
 * and every core node a cut-out graph uses. BiRefNet first.
 */
export function chooseBackgroundRemovalModel(info: ObjectInfo): string | undefined {
  if (!MATTING_CLASSES.every((c) => hasClass(info, c))) return undefined;
  const models = comboOptions(info, "LoadBackgroundRemovalModel", "bg_removal_name");
  return models.find((m) => /birefnet/i.test(m)) ?? models[0];
}

export function summarizeCapabilities(info: ObjectInfo): ServerCapabilities {
  const has = (...names: string[]) => names.every((n) => hasClass(info, n));
  const upscaleModels = comboOptions(info, "UpscaleModelLoader", "model_name");
  const loras = comboOptions(info, "LoraLoader", "lora_name");
  const checkpoints = comboOptions(info, "CheckpointLoaderSimple", "ckpt_name");
  const modelSets = installedModelSets(info);
  const ckptTxt2img = has("CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage") && checkpoints.length > 0;
  const txt2img = ckptTxt2img || modelSets.length > 0;
  const sdxl = checkpoints.find((c) => /(sdxl|sd_xl|xl)/i.test(c)) ?? checkpoints[0];
  const first = modelSets[0];
  const editSet = modelSets.find((m) => m.edit);
  return {
    checkpoints,
    diffusionModels: comboOptions(info, "UNETLoader", "unet_name"),
    modelSets,
    defaultModel: first ? { family: first.family, model: first.model } : ckptTxt2img && sdxl ? { family: "checkpoint", model: sdxl } : null,
    editModel: editSet ? { family: editSet.family, model: editSet.model } : null,
    backgroundRemoval: ((model) => (model ? { model } : null))(chooseBackgroundRemovalModel(info)),
    loras,
    upscaleModels,
    samplers: comboOptions(info, "KSampler", "sampler_name"),
    schedulers: comboOptions(info, "KSampler", "scheduler"),
    nodeCount: Object.keys(info).length,
    supports: {
      txt2img,
      // img2img, inpaint and LoRAs on the checkpoint path; split families edit with instructions instead.
      img2img: ckptTxt2img && has("LoadImage", "VAEEncode"),
      inpaint: ckptTxt2img && has("LoadImage", "VAEEncode", "SetLatentNoiseMask") && (has("LoadImageMask") || has("ImageToMask")),
      edit: !!editSet,
      lora: ckptTxt2img && has("LoraLoader") && loras.length > 0,
      latentUpscale: ckptTxt2img && has("LatentUpscaleBy"),
      modelUpscale: has("UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "VAEEncode") && upscaleModels.length > 0,
    },
  };
}
