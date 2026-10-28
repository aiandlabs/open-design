import type { ObjectInfo, ObjectInfoNode } from "../../src/types.js";

// A trimmed `/object_info` with the node classes this package uses, in the
// shapes current ComfyUI reports: legacy `[[options]]` COMBOs for most enums
// and the newer `["COMBO", { options }]` form for a couple of them.

export const SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "ddim"];
export const SCHEDULERS = ["normal", "karras", "exponential", "simple"];

function node(name: string, required: Record<string, unknown[]>, output: string[], extra: Partial<ObjectInfoNode> = {}): ObjectInfoNode {
  return {
    input: { required: required as ObjectInfoNode["input"]["required"] },
    input_order: { required: Object.keys(required) },
    output,
    output_name: output,
    name,
    display_name: name,
    category: "test",
    output_node: false,
    ...extra,
  };
}

export function makeObjectInfo(opts: { checkpoints?: string[]; loras?: string[]; upscaleModels?: string[] } = {}): ObjectInfo {
  const checkpoints = opts.checkpoints ?? ["sd_xl_base_1.0.safetensors", "v1-5-pruned-emaonly.safetensors"];
  const loras = opts.loras ?? ["detail-tweaker.safetensors"];
  const upscaleModels = opts.upscaleModels ?? [];
  const sampling = {
    seed: ["INT", { default: 0, min: 0, max: 18446744073709551615, control_after_generate: true }],
    steps: ["INT", { default: 20, min: 1, max: 10000 }],
    cfg: ["FLOAT", { default: 8, min: 0, max: 100, step: 0.1 }],
    sampler_name: [SAMPLERS, {}],
    scheduler: [SCHEDULERS, {}],
  };
  return {
    CheckpointLoaderSimple: node("CheckpointLoaderSimple", { ckpt_name: [checkpoints, {}] }, ["MODEL", "CLIP", "VAE"]),
    CLIPTextEncode: node("CLIPTextEncode", { text: ["STRING", { multiline: true }], clip: ["CLIP", {}] }, ["CONDITIONING"]),
    EmptyLatentImage: node(
      "EmptyLatentImage",
      { width: ["INT", { default: 512, min: 16, max: 16384, step: 8 }], height: ["INT", { default: 512, min: 16, max: 16384, step: 8 }], batch_size: ["INT", { default: 1, min: 1, max: 4096 }] },
      ["LATENT"],
    ),
    KSampler: node(
      "KSampler",
      {
        model: ["MODEL", {}],
        ...sampling,
        positive: ["CONDITIONING", {}],
        negative: ["CONDITIONING", {}],
        latent_image: ["LATENT", {}],
        denoise: ["FLOAT", { default: 1, min: 0, max: 1, step: 0.01 }],
      },
      ["LATENT"],
    ),
    KSamplerAdvanced: node(
      "KSamplerAdvanced",
      {
        model: ["MODEL", {}],
        add_noise: [["enable", "disable"], {}],
        noise_seed: ["INT", { default: 0, min: 0, max: 18446744073709551615, control_after_generate: true }],
        steps: ["INT", { default: 20, min: 1, max: 10000 }],
        cfg: ["FLOAT", { default: 8, min: 0, max: 100 }],
        sampler_name: [SAMPLERS, {}],
        scheduler: [SCHEDULERS, {}],
        positive: ["CONDITIONING", {}],
        negative: ["CONDITIONING", {}],
        latent_image: ["LATENT", {}],
        start_at_step: ["INT", { default: 0, min: 0, max: 10000 }],
        end_at_step: ["INT", { default: 10000, min: 0, max: 10000 }],
        return_with_leftover_noise: [["disable", "enable"], {}],
      },
      ["LATENT"],
    ),
    VAEDecode: node("VAEDecode", { samples: ["LATENT", {}], vae: ["VAE", {}] }, ["IMAGE"]),
    VAEEncode: node("VAEEncode", { pixels: ["IMAGE", {}], vae: ["VAE", {}] }, ["LATENT"]),
    SaveImage: node("SaveImage", { images: ["IMAGE", {}], filename_prefix: ["STRING", { default: "ComfyUI" }] }, [], { output_node: true }),
    PreviewImage: node("PreviewImage", { images: ["IMAGE", {}] }, [], { output_node: true }),
    LoadImage: node("LoadImage", { image: [["example.png"], { image_upload: true }] }, ["IMAGE", "MASK"]),
    LoadImageMask: node("LoadImageMask", { image: [["example.png"], { image_upload: true }], channel: [["alpha", "red", "green", "blue"], {}] }, ["MASK"]),
    ImageScale: node(
      "ImageScale",
      {
        image: ["IMAGE", {}],
        upscale_method: [["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {}],
        width: ["INT", { default: 512, min: 0, max: 16384 }],
        height: ["INT", { default: 512, min: 0, max: 16384 }],
        crop: [["disabled", "center"], {}],
      },
      ["IMAGE"],
    ),
    MaskToImage: node("MaskToImage", { mask: ["MASK", {}] }, ["IMAGE"]),
    ImageToMask: node("ImageToMask", { image: ["IMAGE", {}], channel: ["COMBO", { options: ["red", "green", "blue", "alpha"] }] }, ["MASK"]),
    SetLatentNoiseMask: node("SetLatentNoiseMask", { samples: ["LATENT", {}], mask: ["MASK", {}] }, ["LATENT"]),
    LatentUpscaleBy: node(
      "LatentUpscaleBy",
      { samples: ["LATENT", {}], upscale_method: [["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], {}], scale_by: ["FLOAT", { default: 1.5, min: 0.01, max: 8 }] },
      ["LATENT"],
    ),
    LoraLoader: node(
      "LoraLoader",
      {
        model: ["MODEL", {}],
        clip: ["CLIP", {}],
        lora_name: [loras, {}],
        strength_model: ["FLOAT", { default: 1, min: -100, max: 100 }],
        strength_clip: ["FLOAT", { default: 1, min: -100, max: 100 }],
      },
      ["MODEL", "CLIP"],
    ),
    UpscaleModelLoader: node("UpscaleModelLoader", { model_name: ["COMBO", { options: upscaleModels }] }, ["UPSCALE_MODEL"]),
    ImageUpscaleWithModel: node("ImageUpscaleWithModel", { upscale_model: ["UPSCALE_MODEL", {}], image: ["IMAGE", {}] }, ["IMAGE"]),
  };
}
