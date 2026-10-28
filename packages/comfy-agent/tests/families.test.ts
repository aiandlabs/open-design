import { describe, expect, it } from "vitest";

import { applyEdit, describeDoc, parseGraphEdit } from "../src/edits.js";
import { GraphEditError } from "../src/errors.js";
import { FAMILY_SPECS, installedSplitModels } from "../src/families.js";
import {
  availableBaseGraphs,
  buildEdit,
  buildTxt2Img,
  chooseEditModel,
  chooseModel,
  defaultSampling,
  detectFamily,
  docFromGraph,
  planGenerationSize,
  type ModelChoice,
} from "../src/graphs.js";
import { summarizeCapabilities } from "../src/object-info.js";
import { composeKeyedPrompt, stripKeyedPrompt } from "../src/prompts.js";
import type { ObjectInfo } from "../src/types.js";
import { validateGraph } from "../src/validate.js";
import { FAMILY_NODES } from "./fixtures/object-info-families.js";
import { makeObjectInfo } from "./fixtures/object-info.js";

/** SDXL fixture plus the real node definitions for the split-loader families. */
function serverInfo(lists?: { unets?: string[]; clips?: string[]; vaes?: string[] }): ObjectInfo {
  const info: ObjectInfo = { ...makeObjectInfo(), ...structuredClone(FAMILY_NODES) };
  const set = (cls: string, input: string, values: string[] | undefined) => {
    if (values) (info[cls]!.input.required![input] as unknown[])[0] = values;
  };
  set("UNETLoader", "unet_name", lists?.unets);
  set("CLIPLoader", "clip_name", lists?.clips);
  set("VAELoader", "vae_name", lists?.vaes);
  return info;
}

const info = serverInfo();

function choose(family: string): ModelChoice {
  const found = summarizeCapabilities(info).modelSets.find((m) => m.family === family)!;
  return { family: found.family, label: found.label, model: found.model, textEncoder: found.textEncoder, vae: found.vae, edit: found.edit };
}

function t2i(family: string, extra: Partial<Parameters<typeof buildTxt2Img>[0]> = {}) {
  const m = choose(family);
  return buildTxt2Img({
    checkpoint: m.model,
    family: m.family,
    textEncoder: m.textEncoder!,
    vae: m.vae!,
    positive: "A single red paper lantern, flat illustration.",
    negative: "text",
    width: 1024,
    height: 1024,
    ...defaultSampling(m.family),
    seed: 7,
    ...extra,
  });
}

describe("model families", () => {
  it("finds every installed split-loader set with the files the real server lists", () => {
    const sets = summarizeCapabilities(info).modelSets;
    expect(sets.map((s) => [s.family, s.model, s.textEncoder, s.vae, s.edit])).toEqual([
      ["zimage", "z_image_turbo_int8_convrot.safetensors", "qwen_3_4b.safetensors", "ae.safetensors", false],
      ["flux2_klein", "flux-2-klein-4b.safetensors", "qwen_3_4b.safetensors", "flux2-vae.safetensors", true],
      ["ming", "ming_image_0.1_design_int8_convrot.safetensors", "ming_image_0.1_ling_mini_2.0_w4a8.safetensors", "ming_image_vae_bf16.safetensors", true],
    ]);
  });

  it("detects families from file names", () => {
    expect(detectFamily("z_image_turbo_bf16.safetensors")).toBe("zimage");
    expect(detectFamily("flux-2-klein-4b.safetensors")).toBe("flux2_klein");
    expect(detectFamily("ming_image_0.1_design_int8_convrot.safetensors")).toBe("ming");
    // Non-commercial or differently-sampled variants are not matched.
    expect(detectFamily("flux-2-klein-9b.safetensors")).toBe("unknown");
    expect(detectFamily("flux-2-klein-base-4b.safetensors")).toBe("unknown");
    expect(detectFamily("sd_xl_base_1.0.safetensors")).toBe("sdxl");
  });

  it("prefers int8 files, then bf16, and never picks a set with a missing part", () => {
    const sets = installedSplitModels(
      { diffusionModels: ["z_image_turbo_bf16.safetensors", "z_image_turbo_int8_convrot.safetensors", "flux-2-klein-4b.safetensors"], textEncoders: ["qwen_3_4b.safetensors"], vaes: ["ae.safetensors"] },
      () => true,
    );
    expect(sets.map((s) => [s.family, s.model])).toEqual([["zimage", "z_image_turbo_int8_convrot.safetensors"]]);
  });

  it("prefers the strongest installed family and falls back to SDXL", () => {
    expect(chooseModel(info).family).toBe(FAMILY_SPECS[0]!.family);
    const onlyKlein = serverInfo({ unets: ["flux-2-klein-4b.safetensors"] });
    expect(chooseModel(onlyKlein)).toMatchObject({ family: "flux2_klein", textEncoder: "qwen_3_4b.safetensors", vae: "flux2-vae.safetensors" });
    const none = serverInfo({ unets: [], clips: [], vaes: [] });
    expect(chooseModel(none)).toMatchObject({ family: "sdxl", model: "sd_xl_base_1.0.safetensors" });
    // A named model wins; an unknown one is an error that lists what is installed.
    expect(chooseModel(info, "sd_xl_base_1.0.safetensors").family).toBe("sdxl");
    expect(chooseModel(info, "flux-2-klein-4b.safetensors").family).toBe("flux2_klein");
    expect(() => chooseModel(info, "nope.safetensors")).toThrow(/not installed.*z_image_turbo/);
  });

  it("picks an edit model only when one is installed", () => {
    expect(chooseEditModel(info)?.family).toBe("flux2_klein");
    expect(chooseEditModel(serverInfo({ unets: ["ming_image_0.1_design_int8_convrot.safetensors"] }))?.family).toBe("ming");
    expect(chooseEditModel(serverInfo({ unets: ["z_image_turbo_int8_convrot.safetensors"] }))).toBeUndefined();
  });

  it("reports models and editing in the capability summary and base graphs", () => {
    const caps = summarizeCapabilities(info);
    expect(caps.defaultModel).toEqual({ family: FAMILY_SPECS[0]!.family, model: caps.modelSets[0]!.model });
    expect(caps.editModel).toEqual({ family: "flux2_klein", model: "flux-2-klein-4b.safetensors" });
    expect(caps.supports).toMatchObject({ txt2img: true, edit: true });
    expect(availableBaseGraphs(info)).toEqual(expect.arrayContaining(["txt2img", "img2img", "inpaint", "edit"]));
    const sdxlOnly = summarizeCapabilities(serverInfo({ unets: [], clips: [], vaes: [] }));
    expect(sdxlOnly.supports.edit).toBe(false);
    expect(sdxlOnly.editModel).toBeNull();
    expect(sdxlOnly.defaultModel).toEqual({ family: "checkpoint", model: "sd_xl_base_1.0.safetensors" });
  });

  it("uses the family's official sampler defaults and 1 MP buckets", () => {
    expect(defaultSampling("zimage")).toMatchObject({ steps: 8, cfg: 1, sampler: "res_multistep", scheduler: "simple" });
    expect(defaultSampling("flux2_klein")).toMatchObject({ steps: 4, cfg: 1, sampler: "euler" });
    expect(defaultSampling("ming")).toMatchObject({ steps: 12, sampler: "euler", scheduler: "simple" });
    expect(planGenerationSize("zimage", 1600, 900)).toMatchObject({ width: 1344, height: 768 });
    expect(planGenerationSize("ming", 512, 512)).toMatchObject({ width: 768, height: 768 });
  });
});

describe("split-loader text-to-image graphs", () => {
  it("builds Z-Image-Turbo like image_z_image_turbo.json and validates against the real nodes", () => {
    const doc = t2i("zimage");
    expect(validateGraph(doc.graph, info)).toEqual([]);
    const classes = Object.values(doc.graph).map((n) => n.class_type);
    expect(classes).toEqual(expect.arrayContaining(["UNETLoader", "CLIPLoader", "VAELoader", "ModelSamplingAuraFlow", "ConditioningZeroOut", "EmptySD3LatentImage", "KSampler"]));
    expect(doc.graph.clip!.inputs.type).toBe("lumina2");
    expect(doc.graph.shift!.inputs.shift).toBe(3);
    expect(doc.graph.sampler!.inputs).toMatchObject({ steps: 8, cfg: 1, sampler_name: "res_multistep", scheduler: "simple", model: ["shift", 0] });
    expect(doc.roles.negative).toBeUndefined();
  });

  it("builds FLUX.2 klein and Ming-Image with SamplerCustomAdvanced", () => {
    const klein = t2i("flux2_klein");
    expect(validateGraph(klein.graph, info)).toEqual([]);
    expect(klein.graph.clip!.inputs.type).toBe("flux2");
    expect(klein.graph.sigmas).toMatchObject({ class_type: "Flux2Scheduler", inputs: { steps: 4, width: 1024, height: 1024 } });
    expect(klein.graph.guider).toMatchObject({ class_type: "CFGGuider", inputs: { cfg: 1 } });

    const ming = t2i("ming");
    expect(validateGraph(ming.graph, info)).toEqual([]);
    expect(ming.graph.clip!.inputs.type).toBe("qwen_image");
    expect(ming.graph.shift).toMatchObject({ class_type: "ModelSamplingFlux", inputs: { max_shift: 1.15, base_shift: 0.5 } });
    expect(ming.graph.sigmas).toMatchObject({ class_type: "BasicScheduler", inputs: { steps: 12, scheduler: "simple" } });
    expect(ming.graph.guider!.class_type).toBe("BasicGuider");
  });

  it("refuses a split family without its text encoder or VAE", () => {
    expect(() => buildTxt2Img({ checkpoint: "z_image_turbo_int8_convrot.safetensors", positive: "x", negative: "", width: 1024, height: 1024, ...defaultSampling("zimage") })).toThrow(
      /needs a text encoder and a VAE/,
    );
  });

  it("routes typed edits to the node that owns each setting", () => {
    let d = t2i("flux2_klein");
    d = applyEdit(d, { op: "set_seed", seed: 99 }, info);
    d = applyEdit(d, { op: "set_steps", steps: 6 }, info);
    d = applyEdit(d, { op: "set_cfg", cfg: 2 }, info);
    d = applyEdit(d, { op: "set_sampler", sampler_name: "euler_ancestral" }, info);
    d = applyEdit(d, { op: "set_size", width: 1344, height: 768 }, info);
    d = applyEdit(d, { op: "set_prompt", text: "two lanterns" }, info);
    expect(d.graph.noise!.inputs.noise_seed).toBe(99);
    expect(d.graph.sigmas!.inputs).toMatchObject({ steps: 6, width: 1344, height: 768 });
    expect(d.graph.guider!.inputs.cfg).toBe(2);
    expect(d.graph.select!.inputs.sampler_name).toBe("euler_ancestral");
    expect(d.graph.latent!.inputs).toMatchObject({ width: 1344, height: 768 });
    expect(describeDoc(d)).toMatchObject({ family: "flux2_klein", checkpoint: "flux-2-klein-4b.safetensors", positive: "two lanterns", seed: 99, steps: 6, cfg: 2 });
    expect(validateGraph(d.graph, info)).toEqual([]);

    const ming = applyEdit(t2i("ming"), { op: "set_size", width: 768, height: 1344 }, info);
    expect(ming.graph.shift!.inputs).toMatchObject({ width: 768, height: 1344 });
    expect(() => applyEdit(ming, { op: "set_cfg", cfg: 3 }, info)).toThrow(/no cfg setting/);
    expect(() => applyEdit(t2i("zimage"), { op: "set_negative_prompt", text: "x" }, info)).toThrow(/no negative node/);
    expect(() => applyEdit(t2i("zimage"), { op: "use_img2img", image: "a.png", denoise: 0.5 }, info)).toThrow(/edit_with_instruction/);
    expect(() => applyEdit(t2i("zimage"), { op: "set_checkpoint", ckpt_name: "flux-2-klein-4b.safetensors" }, info)).toThrow(/not a zimage model/);
  });

  it("adds model-only LoRAs to split-loader graphs", () => {
    const withLora = serverInfo();
    (withLora.LoraLoaderModelOnly!.input.required!.lora_name as unknown[])[0] = ["style.safetensors"];
    const d = applyEdit(t2i("zimage"), { op: "add_lora", lora_name: "style.safetensors", strength_model: 0.6 }, withLora);
    expect(d.graph.lora).toMatchObject({ class_type: "LoraLoaderModelOnly", inputs: { model: ["unet", 0], strength_model: 0.6 } });
    expect(d.graph.shift!.inputs.model).toEqual(["lora", 0]);
    expect(validateGraph(d.graph, withLora)).toEqual([]);
  });

  it("re-infers roles from a split-loader graph given as JSON", () => {
    const doc = docFromGraph(structuredClone(t2i("flux2_klein").graph));
    expect(doc.family).toBe("flux2_klein");
    expect(doc.kind).toBe("txt2img");
    expect(doc.roles).toMatchObject({ checkpoint: "unet", positive: "pos", noise: "noise", guider: "guider", sigmas: "sigmas", samplerSelect: "select", latent: "latent", textEncoder: "clip", vae: "vae" });
    // The ConditioningZeroOut negative is not mistaken for a prompt node.
    expect(doc.roles.negative).toBeUndefined();
  });
});

describe("instruction edits", () => {
  it("builds the FLUX.2 klein edit graph like image_flux2_klein_image_edit_4b_distilled.json", () => {
    const m = choose("flux2_klein");
    const doc = buildEdit({ family: "flux2_klein", model: m.model, textEncoder: m.textEncoder!, vae: m.vae!, image: "prev.png", instruction: "Make the background plain white.", seed: 3 });
    expect(doc.kind).toBe("edit");
    expect(validateGraph(doc.graph, info)).toEqual([]);
    expect(doc.graph.refPos).toMatchObject({ class_type: "ReferenceLatent", inputs: { conditioning: ["pos", 0], latent: ["refEncode", 0] } });
    expect(doc.graph.refNeg!.inputs.conditioning).toEqual(["neg", 0]);
    expect(doc.graph.guider!.inputs).toMatchObject({ positive: ["refPos", 0], negative: ["refNeg", 0], cfg: 1 });
    expect(doc.graph.latent!.inputs).toMatchObject({ width: ["refSize", 0], height: ["refSize", 1] });
    expect(docFromGraph(structuredClone(doc.graph))).toMatchObject({ kind: "edit", family: "flux2_klein", roles: { loadImage: "load", positive: "pos" } });
  });

  it("builds the Ming-Image edit graph with an Autogrow image input", () => {
    const m = choose("ming");
    const doc = buildEdit({ family: "ming", model: m.model, textEncoder: m.textEncoder!, vae: m.vae!, image: "prev.png", instruction: "Change the cup to matcha green.", seed: 3 });
    expect(validateGraph(doc.graph, info)).toEqual([]);
    expect(doc.graph.pos).toMatchObject({ class_type: "TextEncodeMingImageEdit", inputs: { prompt: "Change the cup to matcha green.", "images.image_1": ["refScale", 0] } });
    expect(doc.graph.select!.class_type).toBe("SamplerLCM");
    // set_prompt rewrites the instruction on the edit encoder.
    expect(applyEdit(doc, { op: "set_prompt", text: "Remove the steam." }, info).graph.pos!.inputs.prompt).toBe("Remove the steam.");
  });

  it("validates Autogrow inputs: suffixes, link types and inline values", () => {
    const m = choose("ming");
    const g = buildEdit({ family: "ming", model: m.model, textEncoder: m.textEncoder!, vae: m.vae!, image: "p.png", instruction: "x", seed: 1 }).graph;
    g.pos!.inputs["images.image_9"] = ["refScale", 0];
    g.pos!.inputs["images.image_2"] = ["clip", 0];
    g.pos!.inputs["images.image_3"] = "inline";
    const kinds = validateGraph(g, info).map((i) => `${i.input}:${i.kind}`);
    expect(kinds).toEqual(expect.arrayContaining(["images.image_9:bad-link", "images.image_2:link-type", "images.image_3:bad-type"]));
  });

  it("turns any doc into an edit with the installed edit model, keeping the seed", () => {
    const d = applyEdit(t2i("zimage", { seed: 1234 }), { op: "edit_with_instruction", image: "round1.png", instruction: "Remove the extra lanterns." }, info);
    expect(d).toMatchObject({ kind: "edit", family: "flux2_klein" });
    expect(d.graph.load!.inputs.image).toBe("round1.png");
    expect(d.graph.noise!.inputs.noise_seed).toBe(1234);
    expect(() => applyEdit(d, { op: "set_size", width: 512, height: 512 }, info)).toThrow(/takes its size from the input image/);
    const noEditor = serverInfo({ unets: ["z_image_turbo_int8_convrot.safetensors"] });
    expect(() => applyEdit(t2i("zimage"), { op: "edit_with_instruction", image: "a.png", instruction: "x" }, noEditor)).toThrow(GraphEditError);
    expect(parseGraphEdit({ op: "edit_with_instruction", image: "a.png", instruction: "Make it blue" })).toEqual({ op: "edit_with_instruction", image: "a.png", instruction: "Make it blue" });
  });
});

describe("key-colour prompts", () => {
  it("uses CLIP weights for SD checkpoints and a plain sentence for language-model encoders", () => {
    expect(composeKeyedPrompt("a lantern", "#00FF00")).toMatch(/^\(isolated on a plain flat solid bright chroma-key green background:1\.2\), a lantern/);
    const sentence = composeKeyedPrompt("A single red lantern", "#00FF00", "sentences");
    expect(sentence).toMatch(/^A single red lantern\. The subject is isolated on a plain, flat, uniform bright chroma-key green background/);
    expect(sentence).not.toContain(":1.2");
    expect(stripKeyedPrompt(sentence)).toBe("A single red lantern.");
  });
});
