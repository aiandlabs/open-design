import { describe, expect, it } from "vitest";

import { addBackgroundRemoval, buildTxt2Img, chooseBackgroundRemovalModel } from "../src/graphs.js";
import { summarizeCapabilities } from "../src/object-info.js";
import { validateGraph } from "../src/validate.js";
import { FAMILY_NODES } from "./fixtures/object-info-families.js";
import { MATTING_NODES } from "./fixtures/object-info-matting.js";
import { makeObjectInfo } from "./fixtures/object-info.js";

const withMatting = () => ({ ...makeObjectInfo(), ...structuredClone(FAMILY_NODES), ...structuredClone(MATTING_NODES) });

function baseParams(): Parameters<typeof buildTxt2Img>[0] {
  return {
    checkpoint: "flux-2-klein-4b.safetensors",
    family: "flux2_klein",
    textEncoder: "qwen_3_4b.safetensors",
    vae: "flux2-vae.safetensors",
    positive: "a red lantern",
    negative: "",
    width: 512,
    height: 512,
    steps: 4,
    cfg: 1,
    sampler: "euler",
    scheduler: "simple",
    seed: 1,
  };
}

function base() {
  return buildTxt2Img(baseParams());
}

describe("reference images on a FLUX.2 [klein] text-to-image graph", () => {
  it("attaches each reference latent, in order, to both conditionings and keeps the output size", () => {
    const doc = buildTxt2Img({ ...baseParams(), references: ["shape.png", "set-1.png"] });
    const g = doc.graph;
    const guider = g.guider!;
    const chain = (start: [string, number]) => {
      const out: string[] = [];
      let cur = start;
      while (g[cur[0]]!.class_type === "ReferenceLatent") {
        const enc = g[(g[cur[0]]!.inputs.latent as [string, number])[0]]!;
        const scale = g[(enc.inputs.pixels as [string, number])[0]]!;
        out.unshift(String(g[(scale.inputs.image as [string, number])[0]]!.inputs.image));
        cur = g[cur[0]]!.inputs.conditioning as [string, number];
      }
      return { refs: out, base: cur[0] };
    };
    const pos = chain(guider.inputs.positive as [string, number]);
    const neg = chain(guider.inputs.negative as [string, number]);
    expect(pos).toEqual({ refs: ["shape.png", "set-1.png"], base: "pos" });
    expect(neg.refs).toEqual(["shape.png", "set-1.png"]);
    expect(g[neg.base]!.class_type).toBe("ConditioningZeroOut");
    // Text-to-image: the latent is the requested size, not a reference's size.
    expect(g.latent!.inputs).toMatchObject({ width: 512, height: 512 });
    expect(doc.roles.positive).toBe("pos");
    expect(validateGraph(g, withMatting())).toEqual([]);
  });

  it("refuses references for families that cannot condition on images", () => {
    expect(() => buildTxt2Img({ ...baseParams(), checkpoint: "sd_xl_base_1.0.safetensors", family: undefined, references: ["a.png"] })).toThrow(/reference/);
  });
});

describe("background removal (matting)", () => {
  it("finds the installed BiRefNet model only when every core node it needs is present", () => {
    expect(chooseBackgroundRemovalModel(withMatting())).toBe("birefnet.safetensors");
    const noJoin = withMatting();
    delete (noJoin as Record<string, unknown>).JoinImageWithAlpha;
    expect(chooseBackgroundRemovalModel(noJoin)).toBeUndefined();
    expect(chooseBackgroundRemovalModel({ ...makeObjectInfo(), ...structuredClone(FAMILY_NODES) })).toBeUndefined();
    expect(summarizeCapabilities(withMatting()).backgroundRemoval).toEqual({ model: "birefnet.safetensors" });
    expect(summarizeCapabilities(makeObjectInfo()).backgroundRemoval).toBeNull();
  });

  it("saves the render with the matting model's foreground as its alpha channel", () => {
    const doc = base();
    const graph = addBackgroundRemoval(doc.graph, "birefnet.safetensors");
    const save = Object.values(graph).find((n) => n.class_type === "SaveImage")!;
    const [joinId] = save.inputs.images as [string, number];
    const join = graph[joinId]!;
    expect(join.class_type).toBe("JoinImageWithAlpha");
    // JoinImageWithAlpha stores 1 − mask as alpha, and RemoveBackground returns the
    // foreground as 1, so the mask is inverted first.
    const invert = graph[(join.inputs.alpha as [string, number])[0]]!;
    expect(invert.class_type).toBe("InvertMask");
    const remove = graph[(invert.inputs.mask as [string, number])[0]]!;
    expect(remove.class_type).toBe("RemoveBackground");
    expect(remove.inputs.image).toEqual(join.inputs.image);
    expect(graph[(remove.inputs.bg_removal_model as [string, number])[0]]!.inputs.bg_removal_name).toBe("birefnet.safetensors");
    expect(validateGraph(graph, withMatting())).toEqual([]);
    // The input graph is left alone.
    expect(Object.values(doc.graph).some((n) => n.class_type === "RemoveBackground")).toBe(false);
  });
});
