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

export const PLANNER_SYSTEM = `You plan image generation on a local ComfyUI server running Stable Diffusion checkpoints.
You never draw. You write prompts and choose sampling settings, and you turn the user's request into a checklist a vision model can verify by looking at the finished image.

Checklist rules:
- 3 to 7 items. Each item is one visually checkable fact about the image: subject count, subject identity, colours, composition, background, style, absence of an unwanted element.
- Write each item as a statement that is true for a correct image ("Exactly one paper lantern is visible").
- Do not include size, resolution, file format or transparency items; code checks those.
- Use short ids such as "count", "subject", "background", "style".

Prompt rules for Stable Diffusion:
- positive_prompt: comma-separated visual description, most important first. State counts explicitly ("a single", "one", "solo"). No instructions, no negations.
- negative_prompt: things to avoid, comma-separated (for example "multiple objects, duplicates, text, watermark, blurry").
- When the asset needs a transparent background, describe the subject only; the system adds the flat key-colour background phrase itself. Choose key_color from the allowed list so it is far from every colour in the subject.`;

export const REPAIR_SYSTEM = `You repair a Stable Diffusion generation on a ComfyUI server. You receive the request, the checklist, the current graph settings, the critic's grades with repair hints, the code checks, and the image.
Return a small list of typed edits that are most likely to make the failing checklist items pass without breaking the passing ones.

Guidance:
- Prompt problems (wrong count, missing element, extra objects, wrong style): set_prompt and/or set_negative_prompt. Put the failing requirement early in the positive prompt and add the unwanted element to the negative prompt.
- Composition is mostly wrong: change set_seed to a new value as well as fixing the prompt.
- Composition is mostly right but a detail is wrong: img2img_from_previous with denoise 0.45-0.7, or inpaint_previous on the region (x, y, width, height are fractions 0-1 of the image) with denoise 0.6-0.9.
- Prompt adherence is weak: raise cfg slightly (at most 9 for SDXL) or steps.
- Background not flat or not clean for a transparent asset: strengthen the prompt and negative prompt, or set_key_color to a colour further from the subject.
- Only use samplers, schedulers, LoRAs, checkpoints and upscale models from the lists you are given.
- Prefer at most 4 edits. Do not repeat an edit that already failed in an earlier round.`;

export const CRITIC_SYSTEM = `You are a strict visual QA reviewer. You receive one image and a checklist.
Grade each checklist item pass or fail by looking at the image only. Pass an item only when it is clearly true. For every failed item give one short concrete repair hint (what is wrong and what should change).
If the asset is a cut-out, transparent areas are shown as flat mid-grey; judge the subject and the cleanliness of its edges, not the grey.`;

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

export function keyColorPhrase(hex: string): string {
  const name = KEY_COLORS[hex.toUpperCase()] ?? "flat solid";
  return `isolated on a plain flat solid ${name} background, uniform ${name} backdrop, studio cut-out, no shadow, no floor`;
}

export const KEY_NEGATIVE = "gradient background, textured background, scenery, environment, cast shadow, floor, vignette";
