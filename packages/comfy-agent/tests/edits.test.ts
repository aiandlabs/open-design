import { describe, expect, it } from "vitest";

import { applyEdit, applyEdits, currentSize, describeDoc, parseGraphEdit } from "../src/edits.js";
import { GraphEditError } from "../src/errors.js";
import { buildTxt2Img, docFromGraph, type GraphDoc } from "../src/graphs.js";
import { validateGraph } from "../src/validate.js";
import type { ApiWorkflow } from "../src/types.js";
import { makeObjectInfo } from "./fixtures/object-info.js";

const info = makeObjectInfo({ upscaleModels: ["4x-test.pth"] });

function doc(): GraphDoc {
  return buildTxt2Img({
    checkpoint: "sd_xl_base_1.0.safetensors",
    positive: "a lantern",
    negative: "blurry",
    width: 1024,
    height: 1024,
    steps: 20,
    cfg: 7,
    sampler: "euler",
    scheduler: "normal",
    seed: 1,
  });
}

describe("typed edits", () => {
  it("sets prompt, negative, size, steps, cfg, sampler, scheduler, seed, denoise and checkpoint", () => {
    const d = applyEdits(
      doc(),
      [
        { op: "set_prompt", text: "one red lantern" },
        { op: "set_negative_prompt", text: "many lanterns" },
        { op: "set_size", width: 832, height: 1216 },
        { op: "set_steps", steps: 30 },
        { op: "set_cfg", cfg: 5.5 },
        { op: "set_sampler", sampler_name: "dpmpp_2m" },
        { op: "set_scheduler", scheduler: "karras" },
        { op: "set_seed", seed: 99 },
        { op: "set_denoise", denoise: 0.9 },
        { op: "set_checkpoint", ckpt_name: "v1-5-pruned-emaonly.safetensors" },
      ],
      info,
    );
    expect(describeDoc(d)).toMatchObject({
      positive: "one red lantern",
      negative: "many lanterns",
      size: { width: 832, height: 1216 },
      steps: 30,
      cfg: 5.5,
      sampler: "dpmpp_2m",
      scheduler: "karras",
      seed: 99,
      denoise: 0.9,
      checkpoint: "v1-5-pruned-emaonly.safetensors",
    });
    expect(validateGraph(d.graph, info)).toEqual([]);
  });

  it("does not mutate the input document", () => {
    const d = doc();
    const before = JSON.stringify(d);
    applyEdit(d, { op: "set_prompt", text: "changed" }, info);
    expect(JSON.stringify(d)).toBe(before);
  });

  it("rejects samplers, schedulers, checkpoints and LoRAs the server does not have", () => {
    expect(() => applyEdit(doc(), { op: "set_sampler", sampler_name: "made_up" }, info)).toThrow(GraphEditError);
    expect(() => applyEdit(doc(), { op: "set_scheduler", scheduler: "made_up" }, info)).toThrow(/not available/);
    expect(() => applyEdit(doc(), { op: "set_checkpoint", ckpt_name: "x.safetensors" }, info)).toThrow(/not available/);
    expect(() => applyEdit(doc(), { op: "add_lora", lora_name: "x.safetensors" }, info)).toThrow(/not available/);
    expect(() => applyEdit(doc(), { op: "add_lora", lora_name: "detail-tweaker.safetensors" }, makeObjectInfo({ loras: [] }))).toThrow(/none installed/);
  });

  it("rejects out-of-range numbers and bad sizes", () => {
    expect(() => applyEdit(doc(), { op: "set_steps", steps: 0 }, info)).toThrow(/outside/);
    expect(() => applyEdit(doc(), { op: "set_denoise", denoise: 1.5 }, info)).toThrow(/outside/);
    expect(() => applyEdit(doc(), { op: "set_size", width: 1001, height: 1024 }, info)).toThrow(/multiples of 8/);
    expect(() => applyEdit(doc(), { op: "set_steps", steps: 2.5 }, info)).toThrow(/integer/);
  });

  it("adds a LoRA and rewires model and clip consumers through it", () => {
    const d = applyEdit(doc(), { op: "add_lora", lora_name: "detail-tweaker.safetensors", strength_model: 0.6 }, info);
    const g = d.graph;
    expect(d.roles.loras).toEqual(["lora"]);
    expect(g.lora!.inputs).toMatchObject({ model: ["ckpt", 0], clip: ["ckpt", 1], strength_model: 0.6, strength_clip: 0.6 });
    expect(g.sampler!.inputs.model).toEqual(["lora", 0]);
    expect(g.pos!.inputs.clip).toEqual(["lora", 1]);
    expect(g.neg!.inputs.clip).toEqual(["lora", 1]);
    expect(g.decode!.inputs.vae).toEqual(["ckpt", 2]);
    expect(validateGraph(g, info)).toEqual([]);
    expect(() => applyEdit(d, { op: "add_lora", lora_name: "detail-tweaker.safetensors" }, info)).toThrow(/already/);
  });

  it("switches to img2img from an uploaded previous output, keeping settings and LoRAs", () => {
    const withLora = applyEdit(doc(), { op: "add_lora", lora_name: "detail-tweaker.safetensors" }, info);
    const d = applyEdit(withLora, { op: "use_img2img", image: "prev.png", denoise: 0.55 }, info);
    expect(d.kind).toBe("img2img");
    expect(d.graph.load!.inputs.image).toBe("prev.png");
    expect(d.graph.sampler!.inputs).toMatchObject({ latent_image: ["encode", 0], denoise: 0.55, seed: 1, model: ["lora", 0] });
    expect(currentSize(d)).toEqual({ width: 1024, height: 1024 });
    expect(validateGraph(d.graph, info)).toEqual([]);
    // Size edits now drive the input scaler.
    const resized = applyEdit(d, { op: "set_size", width: 768, height: 768 }, info);
    expect(resized.graph.scale!.inputs).toMatchObject({ width: 768, height: 768 });
  });

  it("inpaints a region with a mask", () => {
    const d = applyEdit(doc(), { op: "use_inpaint", image: "prev.png", mask: "mask.png", denoise: 0.8 }, info);
    expect(d.kind).toBe("inpaint");
    expect(d.graph.mask!.inputs).toEqual({ image: "mask.png", channel: "red" });
    expect(d.graph.sampler!.inputs.latent_image).toEqual(["noiseMask", 0]);
    expect(validateGraph(d.graph, info)).toEqual([]);
    const resized = applyEdit(d, { op: "set_size", width: 768, height: 768 }, info);
    expect(resized.graph.maskScale!.inputs).toMatchObject({ width: 768, height: 768 });
  });

  it("adds a latent upscale pass", () => {
    const d = applyEdit(doc(), { op: "upscale", scale_by: 1.5 }, info);
    expect(d.roles.upscale).toEqual(["upLatent", "upSampler", "upDecode"]);
    expect(d.graph.save!.inputs.images).toEqual(["upDecode", 0]);
    expect(d.graph.upSampler!.inputs).toMatchObject({ latent_image: ["upLatent", 0], denoise: 0.45 });
    expect(validateGraph(d.graph, info)).toEqual([]);
    expect(() => applyEdit(d, { op: "upscale", scale_by: 2 }, info)).toThrow(/already/);
  });

  it("adds a model upscale pass only when an upscale model is installed", () => {
    const d = applyEdit(doc(), { op: "upscale", scale_by: 2, method: "model" }, info);
    expect(d.graph.upModel!.inputs.model_name).toBe("4x-test.pth");
    expect(d.graph.upFit!.inputs).toMatchObject({ width: 2048, height: 2048 });
    expect(validateGraph(d.graph, info)).toEqual([]);
    expect(() => applyEdit(doc(), { op: "upscale", scale_by: 2, method: "model" }, makeObjectInfo())).toThrow(/no upscale models/);
  });

  it("edits foreign graphs through inferred roles (KSamplerAdvanced uses noise_seed)", () => {
    const g: ApiWorkflow = {
      "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "sd_xl_base_1.0.safetensors" } },
      "2": { class_type: "CLIPTextEncode", inputs: { text: "old", clip: ["1", 1] } },
      "3": { class_type: "CLIPTextEncode", inputs: { text: "", clip: ["1", 1] } },
      "4": { class_type: "EmptyLatentImage", inputs: { width: 1024, height: 1024, batch_size: 1 } },
      "5": {
        class_type: "KSamplerAdvanced",
        inputs: {
          model: ["1", 0], add_noise: "enable", noise_seed: 5, steps: 20, cfg: 7, sampler_name: "euler", scheduler: "normal",
          positive: ["2", 0], negative: ["3", 0], latent_image: ["4", 0], start_at_step: 0, end_at_step: 10000, return_with_leftover_noise: "disable",
        },
      },
      "6": { class_type: "VAEDecode", inputs: { samples: ["5", 0], vae: ["1", 2] } },
      "7": { class_type: "SaveImage", inputs: { images: ["6", 0], filename_prefix: "x" } },
    };
    const d = applyEdits(docFromGraph(g), [{ op: "set_prompt", text: "new" }, { op: "set_seed", seed: 77 }], info);
    expect(d.graph["2"]!.inputs.text).toBe("new");
    expect(d.graph["5"]!.inputs.noise_seed).toBe(77);
    expect(validateGraph(d.graph, info)).toEqual([]);
  });

  it("parses loose planner edits and rejects unknown ops", () => {
    expect(parseGraphEdit({ op: "set_cfg", cfg: "6.5" })).toEqual({ op: "set_cfg", cfg: 6.5 });
    expect(parseGraphEdit({ op: "add_lora", lora_name: "a", strength_model: 0.5 })).toEqual({ op: "add_lora", lora_name: "a", strength_model: 0.5 });
    expect(() => parseGraphEdit({ op: "delete_everything" })).toThrow(/unknown edit op/);
    expect(() => parseGraphEdit({ op: "set_steps" })).toThrow(/steps must be a number/);
  });
});
