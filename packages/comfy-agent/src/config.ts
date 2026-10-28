// One place that turns env vars plus explicit settings into agent config, so
// the MCP bin and the Open Design daemon resolve the same values the same way.

import type { ApiShape } from "./llm.js";
import { DEFAULT_AIAND_BASE_URL } from "./models.js";

export const DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188";

export type ComfyAgentConfig = {
  comfyUrl: string;
  aiandApiKey?: string;
  aiandBaseUrl: string;
  /** Absent means: pick from the ai& model catalog (see ./models.ts). */
  plannerModel?: string;
  criticModel?: string;
  apiShape: ApiShape;
  plannerReasoningEffort?: string;
  criticReasoningEffort?: string;
};

export type ComfyAgentConfigInput = Partial<ComfyAgentConfig>;

function clean(v: string | undefined | null): string | undefined {
  const t = typeof v === "string" ? v.trim() : "";
  return t ? t : undefined;
}

export function parseApiShape(v: string | undefined): ApiShape | undefined {
  const t = clean(v)?.toLowerCase();
  if (t === "openai" || t === "chat" || t === "chat-completions") return "openai";
  if (t === "anthropic" || t === "messages") return "anthropic";
  return undefined;
}

/**
 * Parse the compact "model settings" string used where a UI offers a single
 * model field: `planner=<id>, critic=<id>, shape=openai|anthropic`. A bare
 * value with no `=` is the planner model.
 */
export function parseModelSettings(value: string | undefined): Partial<Pick<ComfyAgentConfig, "plannerModel" | "criticModel" | "apiShape">> {
  const out: Partial<Pick<ComfyAgentConfig, "plannerModel" | "criticModel" | "apiShape">> = {};
  const text = clean(value);
  if (!text) return out;
  for (const part of text.split(/[;,\n]/)) {
    const p = part.trim();
    if (!p) continue;
    const eq = p.indexOf("=");
    if (eq < 0) {
      out.plannerModel = p;
      continue;
    }
    const key = p.slice(0, eq).trim().toLowerCase();
    const val = p.slice(eq + 1).trim();
    if (!val) continue;
    if (key === "planner") out.plannerModel = val;
    else if (key === "critic") out.criticModel = val;
    else if (key === "shape" || key === "api" || key === "api_shape") {
      const shape = parseApiShape(val);
      if (shape) out.apiShape = shape;
    }
  }
  return out;
}

/**
 * Explicit input wins over env; env wins over defaults. Models have no
 * literal default: an unset model is chosen from the live catalog when the
 * loop starts, and a set one is validated there.
 */
export function resolveComfyAgentConfig(input: ComfyAgentConfigInput = {}, env: NodeJS.ProcessEnv = process.env): ComfyAgentConfig {
  const key = clean(input.aiandApiKey) ?? clean(env.AIAND_API_KEY);
  const plannerEffort = clean(input.plannerReasoningEffort) ?? clean(env.COMFY_AGENT_PLANNER_REASONING);
  const criticEffort = clean(input.criticReasoningEffort) ?? clean(env.COMFY_AGENT_CRITIC_REASONING);
  const planner = clean(input.plannerModel) ?? clean(env.COMFY_AGENT_PLANNER_MODEL);
  const critic = clean(input.criticModel) ?? clean(env.COMFY_AGENT_CRITIC_MODEL);
  return {
    comfyUrl: (clean(input.comfyUrl) ?? clean(env.COMFYUI_URL) ?? DEFAULT_COMFYUI_URL).replace(/\/+$/, ""),
    ...(key ? { aiandApiKey: key } : {}),
    aiandBaseUrl: (clean(input.aiandBaseUrl) ?? clean(env.AIAND_BASE_URL) ?? DEFAULT_AIAND_BASE_URL).replace(/\/+$/, ""),
    ...(planner ? { plannerModel: planner } : {}),
    ...(critic ? { criticModel: critic } : {}),
    apiShape: input.apiShape ?? parseApiShape(env.COMFY_AGENT_API_SHAPE) ?? "openai",
    ...(plannerEffort ? { plannerReasoningEffort: plannerEffort } : {}),
    ...(criticEffort ? { criticReasoningEffort: criticEffort } : {}),
  };
}
