// Typed graph edits. Each edit is checked against `/object_info` when applied,
// so it can only reference nodes and models this server actually has. Edits
// never mutate their input: they return a new GraphDoc.

import { GraphEditError } from "./errors.js";
import { comboOptions, hasClass, inputSpec } from "./object-info.js";
import { isSplitFamily } from "./families.js";
import { buildEdit, buildImg2Img, buildInpaint, chooseEditModel, detectFamily, modelFileOf, type GraphDoc } from "./graphs.js";
import type { ApiWorkflow, ObjectInfo } from "./types.js";
import { isLink } from "./validate.js";

export type GraphEdit =
  | { op: "set_prompt"; text: string }
  | { op: "set_negative_prompt"; text: string }
  | { op: "set_size"; width: number; height: number }
  | { op: "set_steps"; steps: number }
  | { op: "set_cfg"; cfg: number }
  | { op: "set_sampler"; sampler_name: string }
  | { op: "set_scheduler"; scheduler: string }
  | { op: "set_seed"; seed: number }
  | { op: "set_denoise"; denoise: number }
  | { op: "set_checkpoint"; ckpt_name: string }
  | { op: "add_lora"; lora_name: string; strength_model?: number; strength_clip?: number }
  /** Switch to img2img from an image already uploaded to the server's input folder. */
  | { op: "use_img2img"; image: string; denoise: number }
  /** Inpaint `image` where `mask` (uploaded, white = repaint) is set. */
  | { op: "use_inpaint"; image: string; mask: string; denoise: number }
  /** Add a second, upscaled refinement pass. */
  | { op: "upscale"; scale_by: number; method?: "latent" | "model"; model_name?: string; denoise?: number }
  /**
   * Replace the graph with an instruction edit of `image` (uploaded) by the
   * installed edit-capable model, for example "remove the extra lanterns".
   */
  | { op: "edit_with_instruction"; image: string; instruction: string };

export type GraphEditOp = GraphEdit["op"];

export const GRAPH_EDIT_OPS: readonly GraphEditOp[] = [
  "set_prompt",
  "set_negative_prompt",
  "set_size",
  "set_steps",
  "set_cfg",
  "set_sampler",
  "set_scheduler",
  "set_seed",
  "set_denoise",
  "set_checkpoint",
  "add_lora",
  "use_img2img",
  "use_inpaint",
  "upscale",
  "edit_with_instruction",
];

function clone(doc: GraphDoc): GraphDoc {
  return { ...doc, graph: structuredClone(doc.graph), roles: structuredClone(doc.roles) };
}

function need(doc: GraphDoc, role: keyof GraphDoc["roles"], op: string): string {
  const id = doc.roles[role];
  if (typeof id !== "string" || !doc.graph[id]) {
    throw new GraphEditError(op, `this graph has no ${role} node to edit`);
  }
  return id;
}

function checkNumber(info: ObjectInfo, classType: string, input: string, value: number, op: string, integer = false): void {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new GraphEditError(op, `${input} must be a number`);
  if (integer && !Number.isInteger(value)) throw new GraphEditError(op, `${input} must be an integer`);
  const spec = inputSpec(info, classType, input);
  if (!spec) throw new GraphEditError(op, `${classType} has no input "${input}" on this server`);
  if ((spec.min !== undefined && value < spec.min) || (spec.max !== undefined && value > spec.max)) {
    throw new GraphEditError(op, `${input}=${value} is outside [${spec.min ?? "-inf"}, ${spec.max ?? "inf"}]`);
  }
}

function checkCombo(info: ObjectInfo, classType: string, input: string, value: string, op: string): void {
  const options = comboOptions(info, classType, input);
  if (!options.includes(value)) {
    throw new GraphEditError(op, `${input} "${value}" is not available on this server${options.length ? ` (available: ${options.slice(0, 40).join(", ")})` : " (none installed)"}`);
  }
}

function requireClasses(info: ObjectInfo, op: string, ...classes: string[]): void {
  const missing = classes.filter((c) => !hasClass(info, c));
  if (missing.length) throw new GraphEditError(op, `server is missing node class(es) ${missing.join(", ")}`);
}

function seedInput(classType: string): string {
  return classType === "KSamplerAdvanced" ? "noise_seed" : "seed";
}

/** Width/height owner: EmptyLatentImage for txt2img, the input ImageScale for img2img/inpaint. */
function sizeOwner(doc: GraphDoc): string | undefined {
  return doc.roles.latent ?? doc.roles.inputScale;
}

export function currentSize(doc: GraphDoc): { width: number; height: number } | undefined {
  const id = sizeOwner(doc);
  const inputs = id ? doc.graph[id]?.inputs : undefined;
  if (!inputs || typeof inputs.width !== "number" || typeof inputs.height !== "number") return undefined;
  return { width: inputs.width, height: inputs.height };
}

type SamplingParam = "steps" | "cfg" | "sampler_name" | "scheduler" | "seed" | "denoise";

/**
 * The node and input that hold a sampling parameter. KSampler-style graphs
 * keep everything on the sampler; SamplerCustomAdvanced graphs spread it over
 * RandomNoise (seed), the guider (cfg), the sigmas node (steps, scheduler,
 * denoise) and the sampler-select node. Returns undefined when this graph has
 * no such setting (for example cfg on a BasicGuider).
 */
function samplingTarget(doc: GraphDoc, param: SamplingParam): { id: string; input: string } | undefined {
  const g = doc.graph;
  const sampler = doc.roles.sampler ? g[doc.roles.sampler] : undefined;
  if (!sampler) return undefined;
  if (sampler.class_type !== "SamplerCustomAdvanced") {
    const input = param === "seed" ? seedInput(sampler.class_type) : param;
    return input in sampler.inputs ? { id: doc.roles.sampler!, input } : undefined;
  }
  const at = (role: "noise" | "guider" | "sigmas" | "samplerSelect", input: string) => {
    const id = doc.roles[role];
    return id && g[id] && input in g[id]!.inputs && !isLink(g[id]!.inputs[input]) ? { id, input } : undefined;
  };
  switch (param) {
    case "seed":
      return at("noise", "noise_seed");
    case "cfg":
      return at("guider", "cfg");
    case "steps":
      return at("sigmas", "steps");
    case "scheduler":
      return at("sigmas", "scheduler");
    case "denoise":
      return at("sigmas", "denoise");
    case "sampler_name":
      return at("samplerSelect", "sampler_name");
  }
}

function samplingValue(doc: GraphDoc, param: SamplingParam): unknown {
  const t = samplingTarget(doc, param);
  return t ? doc.graph[t.id]!.inputs[t.input] : undefined;
}

function textOf(node: { inputs: Record<string, unknown> } | undefined): unknown {
  return node?.inputs.text ?? node?.inputs.prompt;
}

/** Sampling and text state of a doc, for traces and for the planner prompt. */
export function describeDoc(doc: GraphDoc): Record<string, unknown> {
  const g = doc.graph;
  return {
    kind: doc.kind,
    family: doc.family,
    checkpoint: modelFileOf(doc) || undefined,
    positive: doc.roles.positive ? textOf(g[doc.roles.positive]) : undefined,
    negative: doc.roles.negative ? textOf(g[doc.roles.negative]) : undefined,
    size: currentSize(doc),
    steps: samplingValue(doc, "steps"),
    cfg: samplingValue(doc, "cfg"),
    sampler: samplingValue(doc, "sampler_name") ?? (doc.roles.samplerSelect ? g[doc.roles.samplerSelect]?.class_type : undefined),
    scheduler: samplingValue(doc, "scheduler"),
    seed: samplingValue(doc, "seed"),
    denoise: samplingValue(doc, "denoise"),
    loras: (doc.roles.loras ?? []).map((id) => ({ name: g[id]?.inputs.lora_name, strength: g[id]?.inputs.strength_model })),
    upscale: doc.roles.upscale ? true : false,
    ...(doc.kind === "edit" && doc.roles.loadImage ? { editOf: g[doc.roles.loadImage]?.inputs.image } : {}),
  };
}

export function applyEdit(doc: GraphDoc, edit: GraphEdit, info: ObjectInfo): GraphDoc {
  const next = clone(doc);
  const g = next.graph;
  switch (edit.op) {
    case "set_prompt":
    case "set_negative_prompt": {
      const id = need(next, edit.op === "set_prompt" ? "positive" : "negative", edit.op);
      if (typeof edit.text !== "string") throw new GraphEditError(edit.op, "text must be a string");
      const node = g[id]!;
      if ("text" in node.inputs) node.inputs.text = edit.text;
      if ("text_g" in node.inputs) node.inputs.text_g = edit.text;
      if ("text_l" in node.inputs) node.inputs.text_l = edit.text;
      if ("prompt" in node.inputs && !("text" in node.inputs)) node.inputs.prompt = edit.text;
      return next;
    }
    case "set_size": {
      const id = sizeOwner(next);
      if (!id) {
        throw new GraphEditError(edit.op, next.kind === "edit" ? "an edit graph takes its size from the input image" : "this graph has no latent or input-scale node that sets the size");
      }
      const cls = g[id]!.class_type;
      checkNumber(info, cls, "width", edit.width, edit.op, true);
      checkNumber(info, cls, "height", edit.height, edit.op, true);
      if (edit.width % 8 !== 0 || edit.height % 8 !== 0) throw new GraphEditError(edit.op, "width and height must be multiples of 8");
      g[id]!.inputs.width = edit.width;
      g[id]!.inputs.height = edit.height;
      for (const extra of ["maskScale"]) {
        if (g[extra]?.class_type === "ImageScale") {
          g[extra]!.inputs.width = edit.width;
          g[extra]!.inputs.height = edit.height;
        }
      }
      // Resolution-dependent schedule and shift nodes (Flux2Scheduler, ModelSamplingFlux) follow the size.
      for (const node of Object.values(g)) {
        if ((node.class_type === "Flux2Scheduler" || node.class_type === "ModelSamplingFlux") && typeof node.inputs.width === "number") {
          node.inputs.width = edit.width;
          node.inputs.height = edit.height;
        }
      }
      return next;
    }
    case "set_steps":
    case "set_cfg":
    case "set_denoise":
    case "set_seed": {
      need(next, "sampler", edit.op);
      const [param, value, integer] =
        edit.op === "set_steps" ? ["steps", edit.steps, true] as const
          : edit.op === "set_cfg" ? ["cfg", edit.cfg, false] as const
            : edit.op === "set_denoise" ? ["denoise", edit.denoise, false] as const
              : ["seed", edit.seed, true] as const;
      const target = samplingTarget(next, param);
      if (!target) throw new GraphEditError(edit.op, `this ${next.family} graph has no ${param} setting`);
      checkNumber(info, g[target.id]!.class_type, target.input, value, edit.op, integer);
      g[target.id]!.inputs[target.input] = value;
      return next;
    }
    case "set_sampler":
    case "set_scheduler": {
      need(next, "sampler", edit.op);
      const param = edit.op === "set_sampler" ? "sampler_name" : "scheduler";
      const value = edit.op === "set_sampler" ? edit.sampler_name : edit.scheduler;
      const target = samplingTarget(next, param);
      if (!target) throw new GraphEditError(edit.op, `this ${next.family} graph has no ${param === "sampler_name" ? "sampler" : "scheduler"} setting`);
      const cls = g[target.id]!.class_type;
      checkCombo(info, cls, target.input, value, edit.op);
      g[target.id]!.inputs[target.input] = value;
      for (const extra of next.roles.upscale ?? []) {
        if (g[extra]?.class_type === cls) g[extra]!.inputs[target.input] = value;
      }
      return next;
    }
    case "set_checkpoint": {
      const id = need(next, "checkpoint", edit.op);
      const cls = g[id]!.class_type;
      const input = "unet_name" in g[id]!.inputs ? "unet_name" : "ckpt_name";
      checkCombo(info, cls, input, edit.ckpt_name, edit.op);
      if (input === "unet_name" && detectFamily(edit.ckpt_name) !== next.family) {
        throw new GraphEditError(edit.op, `${edit.ckpt_name} is not a ${next.family} model; this graph's text encoder and VAE belong to ${next.family}`);
      }
      g[id]!.inputs[input] = edit.ckpt_name;
      return next;
    }
    case "add_lora": {
      const ckpt = need(next, "checkpoint", edit.op);
      // Single-file checkpoints patch model and CLIP; split-loader models take a model-only LoRA.
      const modelOnly = g[ckpt]!.class_type !== "CheckpointLoaderSimple";
      const cls = modelOnly ? "LoraLoaderModelOnly" : "LoraLoader";
      requireClasses(info, edit.op, cls);
      checkCombo(info, cls, "lora_name", edit.lora_name, edit.op);
      const strengthModel = edit.strength_model ?? 0.8;
      const strengthClip = edit.strength_clip ?? strengthModel;
      checkNumber(info, cls, "strength_model", strengthModel, edit.op);
      if (!modelOnly) checkNumber(info, cls, "strength_clip", strengthClip, edit.op);
      const loraIds = next.roles.loras ?? [];
      if (loraIds.some((lid) => g[lid]?.inputs.lora_name === edit.lora_name)) {
        throw new GraphEditError(edit.op, `LoRA ${edit.lora_name} is already applied`);
      }
      // Chain after the last LoRA (or the loader) and re-point every consumer.
      const prev = loraIds.length ? loraIds[loraIds.length - 1]! : ckpt;
      const id = uniqueId(g, "lora");
      g[id] = modelOnly
        ? { class_type: cls, inputs: { model: [prev, 0], lora_name: edit.lora_name, strength_model: strengthModel } }
        : { class_type: cls, inputs: { model: [prev, 0], clip: [prev, 1], lora_name: edit.lora_name, strength_model: strengthModel, strength_clip: strengthClip } };
      rewireConsumers(g, id, prev, modelOnly ? [0] : [0, 1]);
      next.roles.loras = [...loraIds, id];
      return next;
    }
    case "edit_with_instruction": {
      if (typeof edit.instruction !== "string" || !edit.instruction.trim()) throw new GraphEditError(edit.op, "instruction must be a non-empty string");
      const model = chooseEditModel(info);
      if (!model || !isSplitFamily(model.family) || !model.textEncoder || !model.vae) {
        throw new GraphEditError(edit.op, "no instruction-editing model is installed on this server");
      }
      requireClasses(info, edit.op, "LoadImage");
      const seed = Number(describeDoc(next).seed ?? 0);
      return buildEdit({
        family: model.family,
        model: model.model,
        textEncoder: model.textEncoder,
        vae: model.vae,
        image: edit.image,
        instruction: edit.instruction.trim(),
        seed: Number.isFinite(seed) ? seed : 0,
        filenamePrefix: String(g[next.roles.save ?? ""]?.inputs.filename_prefix ?? "comfy-agent"),
      });
    }
    case "use_img2img":
    case "use_inpaint": {
      if (isSplitFamily(next.family) || next.kind === "edit") {
        throw new GraphEditError(edit.op, `${next.family} graphs do not support img2img or inpaint here; use edit_with_instruction`);
      }
      requireClasses(info, edit.op, "LoadImage", "ImageScale", "VAEEncode");
      if (edit.op === "use_inpaint") requireClasses(info, edit.op, "LoadImageMask", "MaskToImage", "ImageToMask", "SetLatentNoiseMask");
      checkNumber(info, g[need(next, "sampler", edit.op)]!.class_type, "denoise", edit.denoise, edit.op);
      const d = describeDoc(next);
      const size = currentSize(next);
      if (!size) throw new GraphEditError(edit.op, "cannot determine the current generation size");
      if (!next.roles.positive || !next.roles.negative || !next.roles.checkpoint) {
        throw new GraphEditError(edit.op, "img2img/inpaint rebuild needs positive, negative and checkpoint nodes");
      }
      const base = {
        checkpoint: String(d.checkpoint),
        positive: String(d.positive ?? ""),
        negative: String(d.negative ?? ""),
        width: size.width,
        height: size.height,
        steps: Number(d.steps),
        cfg: Number(d.cfg),
        sampler: String(d.sampler),
        scheduler: String(d.scheduler),
        seed: Number(d.seed),
        denoise: edit.denoise,
        filenamePrefix: String(g[next.roles.save ?? ""]?.inputs.filename_prefix ?? "comfy-agent"),
      };
      let rebuilt = edit.op === "use_inpaint"
        ? buildInpaint({ ...base, image: edit.image, mask: edit.mask })
        : buildImg2Img({ ...base, image: edit.image });
      // Carry LoRAs over; upscaling is a separate later edit.
      for (const lid of next.roles.loras ?? []) {
        const l = g[lid]!.inputs;
        rebuilt = applyEdit(rebuilt, { op: "add_lora", lora_name: String(l.lora_name), strength_model: Number(l.strength_model), strength_clip: Number(l.strength_clip) }, info);
      }
      return rebuilt;
    }
    case "upscale": {
      if (next.roles.upscale) throw new GraphEditError(edit.op, "graph already has an upscale pass");
      if (isSplitFamily(next.family) || next.kind === "edit") {
        throw new GraphEditError(edit.op, `${next.family} graphs have no second-pass upscale here; generate at the needed size instead`);
      }
      const sampler = need(next, "sampler", edit.op);
      const decode = need(next, "decode", edit.op);
      const ckpt = need(next, "checkpoint", edit.op);
      if (!(edit.scale_by > 1 && edit.scale_by <= 4)) throw new GraphEditError(edit.op, "scale_by must be in (1, 4]");
      const denoise = edit.denoise ?? 0.45;
      const s = g[sampler]!.inputs;
      const lastModel = (next.roles.loras ?? []).at(-1) ?? ckpt;
      const method = edit.method ?? (edit.model_name ? "model" : "latent");
      const ids: string[] = [];
      let latentForPass2: [string, number];
      if (method === "model") {
        requireClasses(info, edit.op, "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale", "VAEEncode");
        const models = comboOptions(info, "UpscaleModelLoader", "model_name");
        const modelName = edit.model_name ?? models[0];
        if (!modelName) throw new GraphEditError(edit.op, "no upscale models installed; use method \"latent\"");
        checkCombo(info, "UpscaleModelLoader", "model_name", modelName, edit.op);
        const size = currentSize(next);
        if (!size) throw new GraphEditError(edit.op, "cannot determine the current generation size");
        const loader = uniqueId(g, "upModel");
        const up = uniqueId(g, "upImage");
        const fit = uniqueId(g, "upFit");
        const enc = uniqueId(g, "upEncode");
        g[loader] = { class_type: "UpscaleModelLoader", inputs: { model_name: modelName } };
        g[up] = { class_type: "ImageUpscaleWithModel", inputs: { upscale_model: [loader, 0], image: [decode, 0] } };
        g[fit] = {
          class_type: "ImageScale",
          inputs: { image: [up, 0], upscale_method: "lanczos", width: roundTo8(size.width * edit.scale_by), height: roundTo8(size.height * edit.scale_by), crop: "disabled" },
        };
        g[enc] = { class_type: "VAEEncode", inputs: { pixels: [fit, 0], vae: [ckpt, 2] } };
        ids.push(loader, up, fit, enc);
        latentForPass2 = [enc, 0];
      } else {
        requireClasses(info, edit.op, "LatentUpscaleBy");
        const methods = comboOptions(info, "LatentUpscaleBy", "upscale_method");
        const lu = uniqueId(g, "upLatent");
        g[lu] = {
          class_type: "LatentUpscaleBy",
          inputs: { samples: [sampler, 0], upscale_method: methods.includes("bislerp") ? "bislerp" : methods[0] ?? "nearest-exact", scale_by: edit.scale_by },
        };
        ids.push(lu);
        latentForPass2 = [lu, 0];
      }
      const pass2 = uniqueId(g, "upSampler");
      g[pass2] = {
        class_type: "KSampler",
        inputs: {
          model: [lastModel, 0],
          positive: s.positive,
          negative: s.negative,
          latent_image: latentForPass2,
          seed: s.seed ?? s.noise_seed ?? 0,
          steps: Math.max(8, Math.round(Number(s.steps ?? 20) * 0.6)),
          cfg: s.cfg ?? 7,
          sampler_name: s.sampler_name,
          scheduler: s.scheduler,
          denoise,
        },
      };
      const dec2 = uniqueId(g, "upDecode");
      g[dec2] = { class_type: "VAEDecode", inputs: { samples: [pass2, 0], vae: [ckpt, 2] } };
      ids.push(pass2, dec2);
      if (next.roles.save) g[next.roles.save]!.inputs.images = [dec2, 0];
      next.roles.upscale = ids;
      return next;
    }
    default: {
      const unknown = edit as { op?: unknown };
      throw new GraphEditError(String(unknown.op), "unknown edit op");
    }
  }
}

export function applyEdits(doc: GraphDoc, edits: GraphEdit[], info: ObjectInfo): GraphDoc {
  return edits.reduce((d, e) => applyEdit(d, e, info), doc);
}

/** Coerce loosely-typed planner output into a GraphEdit, or explain why not. */
export function parseGraphEdit(raw: unknown): GraphEdit {
  if (!raw || typeof raw !== "object") throw new GraphEditError("?", "edit must be an object");
  const r = raw as Record<string, unknown>;
  const op = String(r.op ?? "");
  const num = (k: string) => {
    const v = r[k];
    const n = typeof v === "string" ? Number(v) : v;
    if (typeof n !== "number" || !Number.isFinite(n)) throw new GraphEditError(op, `${k} must be a number`);
    return n;
  };
  const str = (k: string) => {
    const v = r[k];
    if (typeof v !== "string" || !v) throw new GraphEditError(op, `${k} must be a non-empty string`);
    return v;
  };
  const optNum = (k: string) => (r[k] === undefined || r[k] === null ? undefined : num(k));
  switch (op) {
    case "set_prompt":
    case "set_negative_prompt":
      return { op, text: typeof r.text === "string" ? r.text : str("text") };
    case "set_size":
      return { op, width: num("width"), height: num("height") };
    case "set_steps":
      return { op, steps: num("steps") };
    case "set_cfg":
      return { op, cfg: num("cfg") };
    case "set_sampler":
      return { op, sampler_name: str("sampler_name") };
    case "set_scheduler":
      return { op, scheduler: str("scheduler") };
    case "set_seed":
      return { op, seed: num("seed") };
    case "set_denoise":
      return { op, denoise: num("denoise") };
    case "set_checkpoint":
      return { op, ckpt_name: str("ckpt_name") };
    case "add_lora": {
      const sm = optNum("strength_model");
      const sc = optNum("strength_clip");
      return { op, lora_name: str("lora_name"), ...(sm === undefined ? {} : { strength_model: sm }), ...(sc === undefined ? {} : { strength_clip: sc }) };
    }
    case "use_img2img":
      return { op, image: str("image"), denoise: num("denoise") };
    case "use_inpaint":
      return { op, image: str("image"), mask: str("mask"), denoise: num("denoise") };
    case "edit_with_instruction":
      return { op, image: str("image"), instruction: str("instruction") };
    case "upscale": {
      const method = r.method === "model" || r.method === "latent" ? r.method : undefined;
      const d = optNum("denoise");
      return {
        op,
        scale_by: num("scale_by"),
        ...(method ? { method } : {}),
        ...(typeof r.model_name === "string" && r.model_name ? { model_name: r.model_name } : {}),
        ...(d === undefined ? {} : { denoise: d }),
      };
    }
    default:
      throw new GraphEditError(op || "?", `unknown edit op; expected one of ${GRAPH_EDIT_OPS.join(", ")}`);
  }
}

function uniqueId(graph: ApiWorkflow, base: string): string {
  if (!graph[base]) return base;
  for (let i = 2; ; i += 1) if (!graph[`${base}${i}`]) return `${base}${i}`;
}

/** Point every link that read `from[slot]` at `to[slot]` instead (except `to` itself). */
function rewireConsumers(graph: ApiWorkflow, to: string, from: string, slots: number[]): void {
  for (const [id, node] of Object.entries(graph)) {
    if (id === to) continue;
    for (const [name, value] of Object.entries(node.inputs)) {
      if (isLink(value) && String(value[0]) === from && slots.includes(value[1])) {
        node.inputs[name] = [to, value[1]];
      }
    }
  }
}

function roundTo8(v: number): number {
  return Math.max(8, Math.round(v / 8) * 8);
}
