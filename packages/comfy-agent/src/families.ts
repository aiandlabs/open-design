// Model families the base graphs know how to drive, and how to find an
// installed, complete model set for each from the server's `/object_info`
// file lists. Families that ship as separate diffusion model / text encoder /
// VAE files ("split" loaders) are listed here; SD 1.5 and SDXL single-file
// checkpoints are handled by `graphs.ts` as before.
//
// Every family's nodes, sizes and sampler defaults follow its official
// ComfyUI example workflow from Comfy-Org/workflow_templates (shipped in the
// `comfyui-workflow-templates` package that ComfyUI 0.43.0 pins to 0.11.98).
// This module has no imports so that `object-info.ts` and `graphs.ts` can both
// use it without an import cycle.

export type SplitFamily = "zimage" | "flux2_klein" | "ming";
export type ModelFamily = "sd15" | "sdxl" | SplitFamily | "unknown";

export type FamilySampling = {
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
};

export type FamilySpec = {
  family: SplitFamily;
  label: string;
  /** Official example workflow the graph builders follow. */
  source: string;
  /** Files in models/diffusion_models (UNETLoader) of this family. */
  unet: RegExp;
  /** Files in models/text_encoders (CLIPLoader) and the loader type to use. */
  textEncoder: RegExp;
  clipType: string;
  /** Files in models/vae (VAELoader). */
  vae: RegExp;
  /** Node classes the text-to-image graph needs. */
  nodes: readonly string[];
  /** Node classes the instruction-edit graph needs; absent when the family cannot edit. */
  editNodes?: readonly string[];
  /** Distilled checkpoints: steps and CFG are fixed by the distillation and must not be tuned. */
  distilled: boolean;
  /** False when the negative prompt is zeroed out or the guider takes no negative. */
  usesNegative: boolean;
  sampling: FamilySampling;
  /** Prompt style the text encoder was trained on. */
  promptStyle: "sentences";
};

const LOADERS = ["UNETLoader", "CLIPLoader", "VAELoader", "VAEDecode", "SaveImage"] as const;
const CUSTOM_SAMPLING = ["RandomNoise", "KSamplerSelect", "SamplerCustomAdvanced"] as const;

/**
 * Split-loader families, strongest first. The order is the default
 * preference: the first family with a complete installed set becomes the
 * default text-to-image model. It reflects measured results on a 24 GB Apple
 * silicon Mac (see scripts/comfyui-local/README.md).
 */
export const FAMILY_SPECS: readonly FamilySpec[] = [
  {
    family: "flux2_klein",
    label: "FLUX.2 [klein] 4B",
    // image_flux2_klein_text_to_image.json (distilled branch) and image_flux2_klein_image_edit_4b_distilled.json:
    // CLIPLoader(flux2), EmptyFlux2LatentImage, Flux2Scheduler 4 steps, CFGGuider cfg 1, euler,
    // edits add VAEEncode + ReferenceLatent on both conditionings, reference scaled to 1 MP.
    source: "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux2_klein_image_edit_4b_distilled.json",
    // Only the distilled 4B weights (Apache-2.0); 9B is non-commercial, base needs 20 steps at cfg 5.
    unet: /^flux-2-klein-4b(?!.*base).*\.(safetensors|gguf)$/i,
    textEncoder: /^qwen_3_4b(?!.*fp4).*\.safetensors$/i,
    clipType: "flux2",
    vae: /^flux2-vae.*\.safetensors$/i,
    nodes: [...LOADERS, ...CUSTOM_SAMPLING, "CLIPTextEncode", "ConditioningZeroOut", "EmptyFlux2LatentImage", "Flux2Scheduler", "CFGGuider"],
    editNodes: ["LoadImage", "ImageScaleToTotalPixels", "GetImageSize", "VAEEncode", "ReferenceLatent"],
    distilled: true,
    usesNegative: false,
    sampling: { steps: 4, cfg: 1, sampler: "euler", scheduler: "simple" },
    promptStyle: "sentences",
  },
  {
    family: "zimage",
    label: "Z-Image-Turbo",
    // image_z_image_turbo.json: UNETLoader, CLIPLoader(lumina2), ModelSamplingAuraFlow shift 3,
    // ConditioningZeroOut negative, EmptySD3LatentImage 1024², KSampler 8 steps cfg 1 res_multistep/simple.
    source: "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_z_image_turbo.json",
    unet: /^z[-_]image[-_]turbo.*\.(safetensors|gguf)$/i,
    textEncoder: /^qwen_3_4b(?!.*(fp4|fp8)).*\.safetensors$/i,
    clipType: "lumina2",
    vae: /^ae\.safetensors$/i,
    nodes: [...LOADERS, "ModelSamplingAuraFlow", "CLIPTextEncode", "ConditioningZeroOut", "EmptySD3LatentImage", "KSampler"],
    distilled: true,
    usesNegative: false,
    sampling: { steps: 8, cfg: 1, sampler: "res_multistep", scheduler: "simple" },
    promptStyle: "sentences",
  },
  {
    family: "ming",
    label: "Ming-Image-0.1-Design",
    // image_ming_image_01_design_t2i.json / image_ming_image_01_design_image_edit.json:
    // CLIPLoader(qwen_image) with the Ling-mini-2.0 encoder, ModelSamplingFlux(1.15, 0.5, w, h),
    // BasicScheduler simple 12 steps, BasicGuider (no CFG), euler; edits use TextEncodeMingImageEdit and SamplerLCM.
    source: "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_ming_image_01_design_t2i.json",
    unet: /^ming_image_0\.1_design(?!_layer).*\.safetensors$/i,
    textEncoder: /^ming_image_0\.1_ling_mini_2\.0(?!_layer).*\.safetensors$/i,
    clipType: "qwen_image",
    vae: /^ming_image_vae.*\.safetensors$/i,
    nodes: [...LOADERS, ...CUSTOM_SAMPLING, "CLIPTextEncode", "ModelSamplingFlux", "BasicScheduler", "BasicGuider", "EmptyLatentImage"],
    editNodes: ["LoadImage", "ImageScaleToTotalPixels", "GetImageSize", "TextEncodeMingImageEdit", "SamplerLCM"],
    distilled: true,
    usesNegative: false,
    sampling: { steps: 12, cfg: 1, sampler: "euler", scheduler: "simple" },
    promptStyle: "sentences",
  },
];

export function familySpec(family: ModelFamily): FamilySpec | undefined {
  return FAMILY_SPECS.find((s) => s.family === family);
}

export function isSplitFamily(family: ModelFamily): family is SplitFamily {
  return FAMILY_SPECS.some((s) => s.family === family);
}

/** Family of a diffusion model file name, or undefined when no split family matches. */
export function splitFamilyOf(file: string): SplitFamily | undefined {
  return FAMILY_SPECS.find((s) => s.unet.test(file))?.family;
}

/** One installed, complete model set for a split family. */
export type SplitModelSet = {
  family: SplitFamily;
  label: string;
  model: string;
  textEncoder: string;
  vae: string;
  txt2img: boolean;
  edit: boolean;
};

export type InstalledFiles = {
  diffusionModels: readonly string[];
  textEncoders: readonly string[];
  vaes: readonly string[];
};

/**
 * Every split family with a diffusion model, text encoder and VAE installed
 * and the node classes its graph needs, in preference order. Within a family
 * the int8 file is preferred: on Apple silicon ComfyUI keeps int8 weights
 * quantized and dequantizes per layer, so it needs about half the memory of bf16.
 */
export function installedSplitModels(files: InstalledFiles, hasNodes: (classes: readonly string[]) => boolean): SplitModelSet[] {
  const out: SplitModelSet[] = [];
  for (const spec of FAMILY_SPECS) {
    const model = pickFile(files.diffusionModels.filter((f) => spec.unet.test(f)));
    const textEncoder = pickFile(files.textEncoders.filter((f) => spec.textEncoder.test(f)));
    const vae = pickFile(files.vaes.filter((f) => spec.vae.test(f)));
    if (!model || !textEncoder || !vae) continue;
    const txt2img = hasNodes(spec.nodes);
    const edit = txt2img && !!spec.editNodes && hasNodes(spec.editNodes);
    if (!txt2img) continue;
    out.push({ family: spec.family, label: spec.label, model, textEncoder, vae, txt2img, edit });
  }
  return out;
}

function pickFile(files: string[]): string | undefined {
  const rank = (f: string) => (/int8/i.test(f) ? 0 : /w4a8/i.test(f) ? 1 : /bf16/i.test(f) ? 2 : /fp8|mxfp8|nvfp4|fp4/i.test(f) ? 4 : 3);
  return [...files].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))[0];
}
