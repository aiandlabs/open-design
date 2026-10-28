// The independent judge: a vision model from a different family than the
// planner and the critic grades the task's fixed human checklist. It never
// sees the planner's checklist, the prompt the loop wrote, or which arm made
// the image. Instructions are identical for every arm.

import { cutoutSheet, encodePng, fitWithin, type RgbaImage } from "../../src/image.js";
import type { AiandClient, ApiShape, JsonSchema, StructuredResult } from "../../src/llm.js";
import { selectModel, type AiandModel } from "../../src/models.js";
import { normalizeCriticItems } from "../../src/prompts.js";
import type { JudgeResult, JudgedItem } from "./results.js";
import { checklistIds, type EvalTask } from "./tasks.js";

/**
 * Ranking hints only, like the loop's role preferences: a listed id is used
 * when the live catalog offers it with vision and tool calling, is not
 * deprecated, and is from a family the planner and critic do not use.
 */
export const JUDGE_PREFERENCES: readonly string[] = ["google/gemma-4-31b-it"];
export const JUDGE2_PREFERENCES: readonly string[] = ["qwen/qwen3.8-27b"];
export const JUDGE_CAPABILITIES: readonly string[] = ["vision", "tool_calling"];

/** Model family: the provider the catalog reports, else the id's namespace ("google/gemma-…" → "google"). */
export function modelFamily(id: string, provider?: string): string {
  return (provider?.trim() || (id.includes("/") ? id.slice(0, id.indexOf("/")) : id)).toLowerCase();
}

/**
 * Pick a judge from the live catalog. It must be independent of the models
 * it judges: every model from an excluded family (the planner's, the
 * critic's, and for the second judge the first judge's) is out. A configured
 * id from an excluded family is refused rather than silently replaced.
 */
export function selectJudgeModel(
  models: readonly AiandModel[],
  input: { configured?: string; preferred: readonly string[]; excludeFamilies: readonly string[]; now?: number },
): string {
  const excluded = new Set(input.excludeFamilies.map((f) => f.toLowerCase()));
  const familyOf = (id: string) => modelFamily(id, models.find((m) => m.id === id)?.provider);
  if (input.configured && excluded.has(familyOf(input.configured))) {
    throw new Error(`judge ${input.configured} is from the ${familyOf(input.configured)} family, which a judged or other judge model also uses; pick a model from another family`);
  }
  return selectModel(
    models.filter((m) => !excluded.has(modelFamily(m.id, m.provider))),
    {
      role: "critic",
      capabilities: JUDGE_CAPABILITIES,
      preferred: input.preferred,
      ...(input.configured ? { configured: input.configured } : {}),
      ...(input.now === undefined ? {} : { now: input.now }),
    },
  );
}

export const JUDGE_TOOL = "submit_judgement";

export const JUDGE_SYSTEM = `You are an independent reviewer grading generated design assets for a product team.
You receive one image and a numbered checklist written by a designer. Grade every item pass or fail by looking at the image only.
- Pass an item only when it is clearly true in the image. If it is doubtful, partly true or cannot be seen, fail it.
- Grade each item on its own. Do not reward effort or style that the item does not ask for.
- Text, letters, numbers, logos and watermarks count as present even when they are garbled or illegible.
- For every item give one short sentence of evidence: what you see that decides it.`;

export const JUDGE_SCHEMA: JsonSchema = {
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
        },
        required: ["id", "pass", "evidence"],
      },
    },
  },
  required: ["items"],
};

/** Explanation that goes with a transparent asset. The view is built by {@link judgeView}. */
export const CUTOUT_EXPLANATION =
  "This asset has a transparent background. It is shown twice, side by side: the left panel composites it over plain white, " +
  "the right panel over near-black. The two panels are separated by a thin magenta divider strip, which is not part of the asset. " +
  "The plain white and near-black areas are the transparent background, so treat them as \"no background\". " +
  "Any shadow, glow, haze, coloured fringe or leftover backdrop that is visible around the subject on either panel is part of the asset: " +
  "take it into account for every item it affects.";

/** What the judge is shown: the image itself, or for a cut-out the light/dark sheet, scaled for the payload. */
export function judgeView(img: RgbaImage, transparent: boolean): Buffer {
  return encodePng(fitWithin(transparent ? cutoutSheet(img) : img, transparent ? 1280 : 1024));
}

export function judgeUserText(task: EvalTask): string {
  const ids = checklistIds(task);
  return [
    `The asset was requested at ${task.width}x${task.height} px.`,
    task.transparent ? CUTOUT_EXPLANATION : "",
    "Checklist:",
    ...task.checklist.map((text, i) => `- ${ids[i]}: ${text}`),
    `Grade every item by id (${ids.join(", ")}) and call ${JUDGE_TOOL}.`,
  ].filter(Boolean).join("\n");
}

/** Align model output with the fixed checklist: one grade per item, in order; a missing grade is a fail. */
export function alignJudgement(task: EvalTask, raw: unknown): { items: JudgedItem[]; ungraded: string[] } {
  const ids = checklistIds(task);
  const graded = normalizeCriticItems(raw);
  const ungraded: string[] = [];
  const items = ids.map((id, i) => {
    const found = graded.find((g) => g.id === id) ?? (graded.length === ids.length && !graded.some((g) => ids.includes(g.id)) ? graded[i] : undefined);
    if (!found) {
      ungraded.push(id);
      return { id, text: task.checklist[i]!, pass: false, evidence: "not graded" };
    }
    return { id, text: task.checklist[i]!, pass: found.pass, ...(found.evidence ? { evidence: found.evidence } : {}) };
  });
  return { items, ungraded };
}

export type JudgeOptions = {
  llm: AiandClient;
  model: string;
  shape: ApiShape;
  /** Retries after an HTTP or format error (default 2). */
  httpRetries?: number;
  retryDelayMs?: number;
  reasoningEffort?: string;
};

/**
 * Grade one image. The same robustness steps as the loop's critic: tolerant
 * normalisation of the item list, and one follow-up request naming every id
 * when grades are missing. Transport errors are retried with backoff; a
 * judge that still fails returns a result with `error` set.
 */
export async function judgeImage(task: EvalTask, view: Buffer, options: JudgeOptions): Promise<JudgeResult> {
  const started = Date.now();
  const ids = checklistIds(task);
  const totals = { inputTokens: 0, outputTokens: 0, costUsd: 0, retries: 0 };
  const call = async (reminder: string): Promise<StructuredResult<Record<string, unknown>>> => {
    let lastErr: unknown;
    const attempts = 1 + (options.httpRetries ?? 2);
    for (let a = 0; a < attempts; a += 1) {
      if (a > 0) {
        totals.retries += 1;
        await new Promise((r) => setTimeout(r, (options.retryDelayMs ?? 3000) * a));
      }
      try {
        const res = await options.llm.structured<Record<string, unknown>>(
          {
            model: options.model,
            system: JUDGE_SYSTEM,
            content: [
              { type: "image", mime: "image/png", base64: view.toString("base64") },
              { type: "text", text: judgeUserText(task) + reminder },
            ],
            schema: JUDGE_SCHEMA,
            toolName: JUDGE_TOOL,
            maxTokens: 4096,
            ...(options.reasoningEffort ? { reasoningEffort: options.reasoningEffort } : {}),
          },
          options.shape,
        );
        totals.inputTokens += res.usage.inputTokens;
        totals.outputTokens += res.usage.outputTokens;
        totals.costUsd += res.reportedCostUsd ?? res.estimatedCostUsd ?? 0;
        return res;
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr;
  };
  const base = { model: options.model, shape: options.shape };
  try {
    let res = await call("");
    let aligned = alignJudgement(task, res.data);
    if (aligned.ungraded.length > 0) {
      totals.retries += 1;
      const again = await call(`\nReturn an "items" array with exactly one entry for each of these ids: ${ids.join(", ")}. "pass" must be a boolean.`);
      const second = alignJudgement(task, again.data);
      if (second.ungraded.length < aligned.ungraded.length) {
        res = again;
        aligned = second;
      }
    }
    return {
      ...base,
      items: aligned.items,
      ...(aligned.ungraded.length ? { ungraded: aligned.ungraded } : {}),
      ...(aligned.ungraded.length === ids.length ? { error: "the judge returned no usable grades" } : {}),
      retries: totals.retries,
      inputTokens: totals.inputTokens,
      outputTokens: totals.outputTokens,
      costUsd: totals.costUsd,
      latencyMs: Date.now() - started,
    };
  } catch (err) {
    return {
      ...base,
      items: [],
      error: err instanceof Error ? `${err.name}: ${err.message}`.slice(0, 500) : String(err),
      retries: totals.retries,
      inputTokens: totals.inputTokens,
      outputTokens: totals.outputTokens,
      costUsd: totals.costUsd,
      latencyMs: Date.now() - started,
    };
  }
}
