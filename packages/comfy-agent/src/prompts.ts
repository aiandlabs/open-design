// Prompts and JSON schemas for the planner and the critic. The models only
// produce JSON through a forced tool call; every value they return is checked
// in code before it touches a graph.

import type { JsonSchema } from "./llm.js";

/** Key colours the loop knows how to describe to a diffusion model. */
export const KEY_COLORS: Record<string, string> = {
  "#00FF00": "bright chroma-key green",
  "#FF00FF": "bright magenta",
  "#0000FF": "pure saturated blue",
  "#00FFFF": "bright cyan",
};

export const PLAN_TOOL = "submit_plan";
export const REPAIR_TOOL = "submit_edits";
export const CRITIC_TOOL = "submit_grades";

export const PLANNER_SYSTEM = `You plan image generation on a local ComfyUI server. The user message names the model family in use.
You never draw. You write prompts and choose sampling settings, and you turn the user's request into a checklist a vision model can verify by looking at the finished image.

Checklist rules:
- 3 to 8 items. Each item is one visually checkable fact about the image: subject count, subject identity, colours, composition, background, style, absence of an unwanted element.
- Write each item as a statement that is true for a correct image ("Exactly one paper lantern is visible").
- Every explicit requirement in the request (a count, a colour, lighting or time of day, an empty area for text, an absence such as "no text" or "no people") gets its own item, worded as close to the request as possible. Do not drop, soften or merge them; add your own items only after these.
- Do not include size, resolution, file format or transparency items; code checks those.
- For a transparent asset, grade the cut-out subject only. Never write an item about the key colour or the generation background: it is removed before grading.
- Use short ids such as "count", "subject", "background", "style".

Prompt rules depend on the model family:
- Stable Diffusion checkpoints (sd15, sdxl): positive_prompt is a comma-separated visual description, most important first. State counts explicitly ("a single", "one", "solo"). No instructions, no negations. negative_prompt lists things to avoid, comma-separated (for example "multiple objects, duplicates, text, watermark, blurry").
- Models with a language-model text encoder (zimage, flux2_klein, ming): positive_prompt is one to three plain descriptive sentences, the subject and its count first, then style, colours, composition and lighting. These models read sentences and follow counts and stated absences well ("The background is empty", "There is no text"). They are distilled and ignore the negative prompt: still fill negative_prompt, but put every requirement in the positive prompt. Do not set steps, cfg, sampler_name or scheduler for these families; their values are fixed.
- When the asset needs a transparent background, describe the subject only; the system adds the flat key-colour background phrase itself. Choose key_color from the allowed list so it is far from every colour in the subject.`;

export const REPAIR_SYSTEM = `You repair an image generation on a local ComfyUI server. You receive the request, the checklist, the current graph settings, the critic's grades with repair hints, the code checks, the image, and which edits this server supports.
Return a small list of typed edits that are most likely to make the failing checklist items pass without breaking the passing ones.

Guidance:
- Prompt problems (wrong count, missing element, extra objects, wrong style): set_prompt and/or set_negative_prompt. Put the failing requirement early in the positive prompt. For sdxl/sd15 add the unwanted element to the negative prompt; for zimage, flux2_klein and ming the negative prompt is ignored, so state the absence in the positive prompt instead.
- Composition is mostly wrong: change set_seed to a new value as well as fixing the prompt.
- Composition is mostly right but a detail is wrong:
  - When instruction editing is available, prefer edit_previous with one short imperative instruction that names only the change and says what stays the same ("Remove the two smaller lanterns; keep the large red lantern and the background unchanged", "Change the cup to matcha green"). It works on the previous image as a whole, so no region is needed.
  - Otherwise use img2img_from_previous with denoise 0.45-0.7, or inpaint_previous on the region (x, y, width, height are fractions 0-1 of the image) with denoise 0.6-0.9. These two exist only for sdxl and sd15.
  - After edit_previous, a set_prompt starts again from text-to-image with the new prompt.
- Prompt adherence is weak on sdxl/sd15: raise cfg slightly (at most 9 for SDXL) or steps. Never change steps, cfg, sampler or scheduler for zimage, flux2_klein or ming: they are distilled and those values are fixed.
- Background not flat or not clean for a transparent asset: strengthen the prompt, or set_key_color to a colour further from the subject, or edit_previous ("Make the background a plain flat bright green with no shadow").
- Only use samplers, schedulers, LoRAs, checkpoints and upscale models from the lists you are given.
- Prefer at most 4 edits. Do not repeat an edit that already failed in an earlier round.`;

export const CRITIC_SYSTEM = `You are a strict visual QA reviewer. You receive one image and a checklist.
Grade each checklist item pass or fail by looking at the image only. Pass an item only when it is clearly true. For every failed item give one short concrete repair hint (what is wrong and what should change).
If the asset is a cut-out, it is shown twice side by side: on plain white (left) and on near-black (right), split by a thin magenta divider that is not part of the asset. The white and near-black areas are the transparent background; judge the subject and the cleanliness of its edges, not those backgrounds.`;

const editOps = [
  "set_prompt",
  "set_negative_prompt",
  "set_steps",
  "set_cfg",
  "set_sampler",
  "set_scheduler",
  "set_seed",
  "set_denoise",
  "set_checkpoint",
  "add_lora",
  "upscale",
  "img2img_from_previous",
  "inpaint_previous",
  "edit_previous",
  "set_key_color",
];

export const PLAN_SCHEMA: JsonSchema = {
  type: "object",
  properties: {
    checklist: {
      type: "array",
      items: {
        type: "object",
        properties: { id: { type: "string" }, text: { type: "string" } },
        required: ["id", "text"],
      },
    },
    positive_prompt: { type: "string" },
    negative_prompt: { type: "string" },
    key_color: { type: "string", enum: Object.keys(KEY_COLORS), description: "Only when transparency is required." },
    steps: { type: "integer" },
    cfg: { type: "number" },
    sampler_name: { type: "string" },
    scheduler: { type: "string" },
    notes: { type: "string" },
  },
  required: ["checklist", "positive_prompt", "negative_prompt"],
};

export const REPAIR_SCHEMA: JsonSchema = {
  type: "object",
  properties: {
    rationale: { type: "string" },
    edits: {
      type: "array",
      items: {
        type: "object",
        properties: {
          op: { type: "string", enum: editOps },
          text: { type: "string" },
          instruction: { type: "string", description: "edit_previous: one short imperative edit instruction." },
          steps: { type: "integer" },
          cfg: { type: "number" },
          sampler_name: { type: "string" },
          scheduler: { type: "string" },
          seed: { type: "integer" },
          denoise: { type: "number" },
          ckpt_name: { type: "string" },
          lora_name: { type: "string" },
          strength_model: { type: "number" },
          strength_clip: { type: "number" },
          scale_by: { type: "number" },
          method: { type: "string", enum: ["latent", "model"] },
          model_name: { type: "string" },
          x: { type: "number" },
          y: { type: "number" },
          width: { type: "number" },
          height: { type: "number" },
          hex: { type: "string", enum: Object.keys(KEY_COLORS) },
        },
        required: ["op"],
      },
    },
  },
  required: ["rationale", "edits"],
};

export const CRITIC_SCHEMA: JsonSchema = {
  type: "object",
  properties: {
    items: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          pass: { type: "boolean" },
          evidence: { type: "string" },
          repair_hint: { type: "string" },
        },
        required: ["id", "pass"],
      },
    },
    overall: { type: "string" },
  },
  required: ["items"],
};

/**
 * Leading, weighted background phrase (ComfyUI `(text:weight)` syntax). It
 * goes first because CLIP attends most to the start of the prompt and long
 * subject prompts push trailing text past the first 77-token chunk.
 */
export function keyColorLead(hex: string): string {
  const name = KEY_COLORS[hex.toUpperCase()] ?? "flat solid";
  return `(isolated on a plain flat solid ${name} background:1.2)`;
}

export function keyColorPhrase(hex: string): string {
  const name = KEY_COLORS[hex.toUpperCase()] ?? "flat solid";
  return `uniform ${name} backdrop, studio cut-out, no shadow, no floor`;
}

/**
 * Sentence form of the backdrop instruction for language-model text encoders
 * (Z-Image, FLUX.2, Ming-Image), which read `(text:1.2)` weights as literal text.
 */
export function keyColorSentence(hex: string): string {
  const name = KEY_COLORS[hex.toUpperCase()] ?? "flat solid";
  return `The subject is isolated on a plain, flat, uniform ${name} background that fills the whole frame, with no shadow, no floor, no gradient and no other objects.`;
}

export type PromptStyle = "tags" | "sentences";

/** Positive prompt for a keyed asset: weighted lead, subject, then the backdrop reminder (or one sentence for LLM encoders). */
export function composeKeyedPrompt(subject: string, hex: string | undefined, style: PromptStyle = "tags"): string {
  if (!hex) return subject;
  if (style === "sentences") return `${subject.replace(/\s+$/, "").replace(/([^.!?])$/, "$1.")} ${keyColorSentence(hex)}`;
  return `${keyColorLead(hex)}, ${subject}, ${keyColorPhrase(hex)}`;
}

/** Remove what {@link composeKeyedPrompt} added, if the planner echoed it back. */
export function stripKeyedPrompt(text: string): string {
  let out = text;
  for (const hex of Object.keys(KEY_COLORS)) {
    out = out.replace(`${keyColorLead(hex)}, `, "").replace(`, ${keyColorPhrase(hex)}`, "").replace(` ${keyColorSentence(hex)}`, "").replace(keyColorSentence(hex), "");
  }
  return out.trim();
}

/**
 * Normalise critic output. Vision models do not always follow the schema:
 * seen in practice are a single item flattened to the top level, `items` as a
 * JSON string, and `pass` as "false"/"pass"/"yes". Unrecognised shapes give
 * an empty list (the caller treats missing grades as ungraded).
 */
export function normalizeCriticItems(raw: unknown): { id: string; pass: boolean; evidence?: string; repair_hint?: string }[] {
  if (!raw || typeof raw !== "object") return [];
  const r = raw as Record<string, unknown>;
  let items: unknown = r.items ?? r.grades ?? r.results;
  if (typeof items === "string") {
    try {
      items = JSON.parse(items);
    } catch {
      items = undefined;
    }
  }
  if (items && !Array.isArray(items) && typeof items === "object") {
    // { count: { pass: true }, ... } or { count: true, ... }
    items = Object.entries(items as Record<string, unknown>).map(([id, v]) =>
      v && typeof v === "object" ? { id, ...(v as object) } : { id, pass: v },
    );
  }
  if (!Array.isArray(items)) items = "id" in r && "pass" in r ? [r] : [];
  const out: { id: string; pass: boolean; evidence?: string; repair_hint?: string }[] = [];
  for (const it of items as unknown[]) {
    if (!it || typeof it !== "object") continue;
    const o = it as Record<string, unknown>;
    const pass = toBool(o.pass ?? o.passed ?? o.result ?? o.verdict);
    if (typeof o.id !== "string" || pass === undefined) continue;
    const evidence = o.evidence ?? o.reason;
    const hint = o.repair_hint ?? o.repairHint ?? o.hint;
    out.push({
      id: o.id,
      pass,
      ...(typeof evidence === "string" && evidence ? { evidence } : {}),
      ...(typeof hint === "string" && hint ? { repair_hint: hint } : {}),
    });
  }
  return out;
}

function toBool(v: unknown): boolean | undefined {
  if (typeof v === "boolean") return v;
  if (typeof v === "string") {
    const t = v.trim().toLowerCase();
    if (["true", "pass", "passed", "yes", "ok"].includes(t)) return true;
    if (["false", "fail", "failed", "no"].includes(t)) return false;
  }
  return undefined;
}

export const KEY_NEGATIVE = "gradient background, textured background, scenery, environment, cast shadow, floor, vignette";
