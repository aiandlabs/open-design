import { describe, expect, it } from "vitest";

import { WorkflowFormatError } from "../src/errors.js";
import {
  buildImg2Img,
  buildInpaint,
  buildTxt2Img,
  detectFamily,
  docFromGraph,
  inferRoles,
  planGenerationSize,
  SDXL_BUCKETS,
} from "../src/graphs.js";
import { parseInputSpec, summarizeCapabilities, widgetOrder } from "../src/object-info.js";
import { convertUiWorkflow, detectWorkflowFormat, toApiWorkflow, type UiWorkflow } from "../src/ui-convert.js";
import { validateGraph } from "../src/validate.js";
import type { ApiWorkflow } from "../src/types.js";
import { makeObjectInfo } from "./fixtures/object-info.js";

const info = makeObjectInfo();
const base = {
  checkpoint: "sd_xl_base_1.0.safetensors",
  positive: "a lantern",
  negative: "blurry",
  width: 1024,
  height: 1024,
  steps: 20,
  cfg: 7,
  sampler: "euler",
  scheduler: "normal",
  seed: 42,
};

describe("object_info parsing", () => {
  it("reads legacy and COMBO-typed enums", () => {
    expect(parseInputSpec("x", [["a", "b"], {}], true)).toMatchObject({ type: "COMBO", options: ["a", "b"], isWidget: true });
    expect(parseInputSpec("x", ["COMBO", { options: ["c"] }], true)).toMatchObject({ type: "COMBO", options: ["c"] });
    expect(parseInputSpec("model", ["MODEL", {}], true)).toMatchObject({ type: "MODEL", isWidget: false });
    expect(parseInputSpec("seed", ["INT", { control_after_generate: true }], true).controlAfterGenerate).toBe(true);
  });

  it("lists widgets in declaration order", () => {
    expect(widgetOrder(info.KSampler!).map((s) => s.name)).toEqual(["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]);
  });

  it("summarizes capabilities", () => {
    const caps = summarizeCapabilities(info);
    expect(caps.supports).toMatchObject({ txt2img: true, img2img: true, inpaint: true, lora: true, latentUpscale: true, modelUpscale: false });
    expect(caps.checkpoints).toContain("sd_xl_base_1.0.safetensors");
  });
});

describe("validateGraph", () => {
  it("accepts every base graph", () => {
    expect(validateGraph(buildTxt2Img(base).graph, info)).toEqual([]);
    expect(validateGraph(buildImg2Img({ ...base, image: "in.png", denoise: 0.5 }).graph, info)).toEqual([]);
    expect(validateGraph(buildInpaint({ ...base, image: "in.png", mask: "mask.png", denoise: 0.8 }).graph, info)).toEqual([]);
  });

  it("reports unknown classes, missing inputs, bad links, link types, enums and ranges", () => {
    const g = buildTxt2Img(base).graph;
    g.extra = { class_type: "NotInstalledNode", inputs: {} };
    delete g.pos!.inputs.text;
    g.neg!.inputs.clip = ["ghost", 0];
    g.decode!.inputs.vae = ["ckpt", 0]; // MODEL into a VAE input
    g.ckpt!.inputs.ckpt_name = "nope.safetensors";
    g.sampler!.inputs.steps = 0;
    g.sampler!.inputs.cfg = "high";
    const kinds = validateGraph(g, info).map((i) => `${i.nodeId}:${i.kind}`);
    expect(kinds).toEqual(
      expect.arrayContaining([
        "extra:unknown-class",
        "pos:missing-input",
        "neg:bad-link",
        "decode:link-type",
        "ckpt:enum-value",
        "sampler:out-of-range",
        "sampler:bad-type",
      ]),
    );
    const enumIssue = validateGraph(g, info).find((i) => i.kind === "enum-value")!;
    expect(enumIssue.allowed).toContain("sd_xl_base_1.0.safetensors");
  });

  it("rejects a value for an enum the server has no entries for", () => {
    const empty = makeObjectInfo({ loras: [] });
    const g: ApiWorkflow = {
      ...buildTxt2Img(base).graph,
      lora: { class_type: "LoraLoader", inputs: { model: ["ckpt", 0], clip: ["ckpt", 1], lora_name: "x.safetensors", strength_model: 1, strength_clip: 1 } },
    };
    expect(validateGraph(g, empty).some((i) => i.kind === "enum-value" && i.nodeId === "lora")).toBe(true);
  });

  it("does not reject LoadImage names that were uploaded after /object_info was read", () => {
    expect(validateGraph(buildImg2Img({ ...base, image: "uploaded-later.png", denoise: 0.5 }).graph, info)).toEqual([]);
  });

  it("flags graphs with no output node", () => {
    const g = buildTxt2Img(base).graph;
    delete g.save;
    expect(validateGraph(g, info).map((i) => i.kind)).toContain("no-output-node");
  });
});

describe("UI → API conversion", () => {
  // A UI-format export: seed with a control_after_generate companion value,
  // a Reroute between checkpoint and decoder, a PrimitiveNode driving the
  // converted `steps` widget, a Note, and a muted preview node.
  const ui = {
    last_node_id: 12,
    last_link_id: 20,
    nodes: [
      { id: 4, type: "CheckpointLoaderSimple", mode: 0, inputs: [], outputs: [{ type: "MODEL", links: [1] }, { type: "CLIP", links: [3, 5] }, { type: "VAE", links: [10] }], widgets_values: ["sd_xl_base_1.0.safetensors"] },
      { id: 6, type: "CLIPTextEncode", mode: 0, inputs: [{ name: "clip", type: "CLIP", link: 3 }], outputs: [{ type: "CONDITIONING", links: [4] }], widgets_values: ["a red lantern"], title: "Positive" },
      { id: 7, type: "CLIPTextEncode", mode: 0, inputs: [{ name: "clip", type: "CLIP", link: 5 }], outputs: [{ type: "CONDITIONING", links: [6] }], widgets_values: ["blurry"] },
      { id: 5, type: "EmptyLatentImage", mode: 0, inputs: [], outputs: [{ type: "LATENT", links: [2] }], widgets_values: [768, 768, 1] },
      {
        id: 3,
        type: "KSampler",
        mode: 0,
        inputs: [
          { name: "model", type: "MODEL", link: 1 },
          { name: "positive", type: "CONDITIONING", link: 4 },
          { name: "negative", type: "CONDITIONING", link: 6 },
          { name: "latent_image", type: "LATENT", link: 2 },
          { name: "steps", type: "INT", link: 15, widget: { name: "steps" } },
        ],
        outputs: [{ type: "LATENT", links: [7] }],
        widgets_values: [123456, "randomize", 20, 6.5, "euler", "karras", 1],
      },
      { id: 11, type: "Reroute", mode: 0, inputs: [{ name: "", type: "*", link: 10 }], outputs: [{ type: "VAE", links: [11] }] },
      { id: 8, type: "VAEDecode", mode: 0, inputs: [{ name: "samples", type: "LATENT", link: 7 }, { name: "vae", type: "VAE", link: 11 }], outputs: [{ type: "IMAGE", links: [9, 12] }] },
      { id: 9, type: "SaveImage", mode: 0, inputs: [{ name: "images", type: "IMAGE", link: 9 }], outputs: [], widgets_values: ["ComfyUI"] },
      { id: 10, type: "PreviewImage", mode: 2, inputs: [{ name: "images", type: "IMAGE", link: 12 }], outputs: [] },
      { id: 12, type: "PrimitiveNode", mode: 0, inputs: [], outputs: [{ type: "INT", links: [15], widget: { name: "steps" } }], widgets_values: [20, "fixed"] },
      { id: 13, type: "Note", mode: 0, inputs: [], outputs: [], widgets_values: ["remember to bump steps"] },
    ],
    links: [
      [1, 4, 0, 3, 0, "MODEL"],
      [2, 5, 0, 3, 3, "LATENT"],
      [3, 4, 1, 6, 0, "CLIP"],
      [4, 6, 0, 3, 1, "CONDITIONING"],
      [5, 4, 1, 7, 0, "CLIP"],
      [6, 7, 0, 3, 2, "CONDITIONING"],
      [7, 3, 0, 8, 0, "LATENT"],
      [9, 8, 0, 9, 0, "IMAGE"],
      [10, 4, 2, 11, 0, "VAE"],
      [11, 11, 0, 8, 1, "VAE"],
      [12, 8, 0, 10, 0, "IMAGE"],
      [15, 12, 0, 3, 4, "INT"],
    ],
    groups: [],
    version: 0.4,
  } as unknown as UiWorkflow;

  it("detects formats", () => {
    expect(detectWorkflowFormat(ui)).toBe("ui");
    expect(detectWorkflowFormat(buildTxt2Img(base).graph)).toBe("api");
    expect(() => detectWorkflowFormat({ hello: 1 })).toThrow(WorkflowFormatError);
  });

  it("converts with /object_info widget order and resolves reroutes", () => {
    const api = convertUiWorkflow(ui, info);
    expect(Object.keys(api).sort()).toEqual(["3", "4", "5", "6", "7", "8", "9"]);
    expect(api["3"]).toEqual({
      class_type: "KSampler",
      inputs: {
        seed: 123456,
        steps: 20,
        cfg: 6.5,
        sampler_name: "euler",
        scheduler: "karras",
        denoise: 1,
        model: ["4", 0],
        positive: ["6", 0],
        negative: ["7", 0],
        latent_image: ["5", 0],
      },
    });
    expect(api["8"]!.inputs.vae).toEqual(["4", 2]);
    expect(api["6"]).toMatchObject({ inputs: { text: "a red lantern", clip: ["4", 1] }, _meta: { title: "Positive" } });
    expect(api["5"]!.inputs).toEqual({ width: 768, height: 768, batch_size: 1 });
    expect(validateGraph(api, info)).toEqual([]);
    // Roles can be inferred on the converted graph.
    expect(inferRoles(api)).toMatchObject({ sampler: "3", positive: "6", negative: "7", latent: "5", checkpoint: "4", save: "9", decode: "8" });
  });

  it("rejects subgraphs with a clear error", () => {
    expect(() => convertUiWorkflow({ ...ui, definitions: { subgraphs: [{ id: "x" }] } }, info)).toThrow(/subgraph/);
    const instance = { ...ui, nodes: [...ui.nodes, { id: 99, type: "0b8f4a52-3b4c-4d5e-8f9a-1b2c3d4e5f60", mode: 0, inputs: [], outputs: [] }] };
    expect(() => toApiWorkflow(instance, info)).toThrow(/subgraph/);
  });

  it("rejects nodes the server does not have", () => {
    const bad = { ...ui, nodes: [...ui.nodes, { id: 50, type: "CustomThing", mode: 0, inputs: [], outputs: [], widgets_values: [] }] };
    expect(() => convertUiWorkflow(bad, info)).toThrow(/CustomThing/);
  });

  it("passes API format through as a copy", () => {
    const g = buildTxt2Img(base).graph;
    const out = toApiWorkflow(g, info);
    expect(out).toEqual(g);
    expect(out).not.toBe(g);
    expect(toApiWorkflow({ prompt: g }, info)).toEqual(g);
  });
});

describe("families and sizes", () => {
  it("detects checkpoint families", () => {
    expect(detectFamily("sd_xl_base_1.0.safetensors")).toBe("sdxl");
    expect(detectFamily("juggernautXL_v9.safetensors")).toBe("sdxl");
    expect(detectFamily("v1-5-pruned-emaonly.safetensors")).toBe("sd15");
    expect(detectFamily("mystery.ckpt")).toBe("unknown");
  });

  it("picks the nearest SDXL aspect bucket", () => {
    expect(planGenerationSize("sdxl", 1920, 1080)).toMatchObject({ width: 1344, height: 768 });
    expect(planGenerationSize("sdxl", 1080, 1920)).toMatchObject({ width: 768, height: 1344 });
    expect(planGenerationSize("sdxl", 2048, 2048)).toMatchObject({ width: 1024, height: 1024 });
    for (const [w, h] of SDXL_BUCKETS) expect(planGenerationSize("sdxl", w, h)).toMatchObject({ width: w, height: h });
  });

  it("scales down for small requests but not below what the family handles", () => {
    expect(planGenerationSize("sdxl", 768, 768)).toMatchObject({ width: 768, height: 768 });
    expect(planGenerationSize("sdxl", 256, 256)).toMatchObject({ width: 768, height: 768 });
    expect(planGenerationSize("sd15", 256, 256)).toMatchObject({ width: 512, height: 512 });
    expect(planGenerationSize("sd15", 1200, 800)).toMatchObject({ width: 768, height: 512 });
  });

  it("builds documents for foreign graphs", () => {
    const doc = docFromGraph(buildImg2Img({ ...base, image: "a.png", denoise: 0.5 }).graph);
    expect(doc.kind).toBe("img2img");
    expect(doc.family).toBe("sdxl");
  });
});
