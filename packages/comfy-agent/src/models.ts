// The ai& model catalog: `GET {base}/models`, parsed defensively and cached
// for a short time, plus the rules that pick a model for each role.
//
// Nothing here treats a literal model id as a default. A role names the
// capabilities it needs; the live list says which models have them. The
// preference lists below only rank models that the live list already offers.

import { createHash } from "node:crypto";

import { LlmHttpError, ModelSelectionError } from "./errors.js";

export const DEFAULT_AIAND_BASE_URL = "https://api.aiand.com/v1";
export const DEFAULT_CATALOG_TTL_MS = 5 * 60_000;

/** One entry of `GET /v1/models`. Fields the API omits stay undefined. */
export type AiandModel = {
  id: string;
  name?: string;
  provider?: string;
  description?: string;
  capabilities: string[];
  contextWindow?: number;
  maxOutputTokens?: number;
  reasoningEfforts: string[];
  reasoningEffortDefault?: string;
  inputPer1m?: number;
  outputPer1m?: number;
  cachedInputPer1m?: number;
  currency?: string;
  /** ISO date string as the API reports it. */
  deprecationDate?: string;
};

/** Kept for callers that only read prices. */
export type ModelPricing = AiandModel;

/** What a model is used for. Each role has its own capability needs. */
export type ModelRole = "planner" | "critic" | "visual-check";

/**
 * Every call the loop and the visual check make goes through
 * `AiandClient.structured`, which forces a tool call, so every role needs
 * `tool_calling`. The critic and the visual check grade images; the planner
 * sees the rendered image when it plans repairs. So every role needs `vision`
 * too. `chat` is not required: the API does not list it for every chat model.
 */
export const ROLE_CAPABILITIES: Readonly<Record<ModelRole, readonly string[]>> = {
  planner: ["vision", "tool_calling"],
  critic: ["vision", "tool_calling"],
  "visual-check": ["vision", "tool_calling"],
};

/**
 * Ranking hints only. These are the models the loop was tested with live.
 * A listed id is used only when the live catalog offers it with the role's
 * capabilities and it is not deprecated; otherwise selection moves on.
 */
export const ROLE_PREFERENCES: Readonly<Record<ModelRole, readonly string[]>> = {
  planner: ["moonshotai/kimi-k3"],
  critic: ["zai-org/glm-5.3-flash"],
  "visual-check": ["zai-org/glm-5.3-flash"],
};

export type ModelCatalogOptions = {
  baseUrl?: string;
  apiKey?: string;
  fetch?: typeof fetch;
  ttlMs?: number;
  requestTimeoutMs?: number;
  /** Clock for the TTL and for deprecation dates. */
  now?: () => number;
};

export type ModelRequirement = {
  role: ModelRole;
  capabilities?: readonly string[];
  preferred?: readonly string[];
};

type CacheEntry = { at: number; pending: Promise<AiandModel[]> };

export class AiandModelCatalog {
  readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch;
  private readonly ttlMs: number;
  private readonly requestTimeoutMs: number;
  private readonly now: () => number;
  private entry: CacheEntry | null = null;

  constructor(options: ModelCatalogOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_AIAND_BASE_URL).replace(/\/+$/, "");
    this.apiKey = options.apiKey?.trim() || undefined;
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.ttlMs = options.ttlMs ?? DEFAULT_CATALOG_TTL_MS;
    this.requestTimeoutMs = options.requestTimeoutMs ?? 30_000;
    this.now = options.now ?? Date.now;
  }

  /** The live list, cached for `ttlMs`. A failed fetch is not cached. */
  list(): Promise<AiandModel[]> {
    const at = this.now();
    if (this.entry && at - this.entry.at < this.ttlMs) return this.entry.pending;
    const pending = this.fetchList();
    const entry: CacheEntry = { at, pending };
    this.entry = entry;
    pending.catch(() => {
      if (this.entry === entry) this.entry = null;
    });
    return pending;
  }

  async get(id: string): Promise<AiandModel | undefined> {
    return (await this.list()).find((m) => m.id === id);
  }

  /** Drop the cached list so the next call fetches again. */
  invalidate(): void {
    this.entry = null;
  }

  /**
   * The model to use for a role. A configured id must exist, must not be
   * deprecated and must have the role's capabilities. With no configured id,
   * the first preferred id that qualifies wins, then the first model in the
   * catalog's own order that qualifies.
   */
  async resolve(requirement: ModelRole | ModelRequirement, configured?: string | null): Promise<string> {
    const req = typeof requirement === "string" ? { role: requirement } : requirement;
    return selectModel(await this.list(), {
      role: req.role,
      capabilities: req.capabilities ?? ROLE_CAPABILITIES[req.role],
      preferred: req.preferred ?? ROLE_PREFERENCES[req.role],
      configured: configured?.trim() || undefined,
      now: this.now(),
    });
  }

  private async fetchList(): Promise<AiandModel[]> {
    const url = `${this.baseUrl}/models`;
    let res: Response;
    try {
      res = await this.fetchImpl(url, {
        headers: this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {},
        signal: AbortSignal.timeout(this.requestTimeoutMs),
      });
    } catch (err) {
      throw new LlmHttpError(url, 0, `request failed: ${err instanceof Error ? err.message : String(err)}`);
    }
    const text = await res.text();
    if (!res.ok) throw new LlmHttpError(url, res.status, text);
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      throw new LlmHttpError(url, res.status, `model list is not JSON: ${text.slice(0, 200)}`);
    }
    return parseModelList(body);
  }
}

// Catalogs made without a custom fetch are shared per base URL and key, so a
// daemon that builds a client per request still fetches the list once per TTL.
const shared = new Map<string, AiandModelCatalog>();

export function sharedModelCatalog(options: { baseUrl?: string; apiKey?: string } = {}): AiandModelCatalog {
  const baseUrl = (options.baseUrl ?? DEFAULT_AIAND_BASE_URL).replace(/\/+$/, "");
  const key = options.apiKey?.trim();
  const id = `${baseUrl}\0${key ? createHash("sha256").update(key).digest("hex").slice(0, 16) : ""}`;
  let catalog = shared.get(id);
  if (!catalog) {
    catalog = new AiandModelCatalog({ baseUrl, ...(key ? { apiKey: key } : {}) });
    shared.set(id, catalog);
  }
  return catalog;
}

export function hasCapabilities(model: AiandModel, required: readonly string[]): boolean {
  return required.every((c) => model.capabilities.includes(c));
}

/** Deprecated once the deprecation date is today or earlier. */
export function isDeprecated(model: AiandModel, now: number = Date.now()): boolean {
  if (!model.deprecationDate) return false;
  const at = Date.parse(model.deprecationDate);
  return Number.isFinite(at) && at <= now;
}

export function selectModel(
  models: readonly AiandModel[],
  input: { role: ModelRole; capabilities: readonly string[]; preferred?: readonly string[]; configured?: string; now?: number },
): string {
  const now = input.now ?? Date.now();
  const usable = models.filter((m) => !isDeprecated(m, now) && hasCapabilities(m, input.capabilities));
  const choices = usable.map((m) => m.id);
  if (input.configured) {
    const model = models.find((m) => m.id === input.configured);
    if (!model) {
      throw new ModelSelectionError({ role: input.role, model: input.configured, reason: "unknown", required: input.capabilities, choices });
    }
    const missing = input.capabilities.filter((c) => !model.capabilities.includes(c));
    if (missing.length > 0) {
      throw new ModelSelectionError({ role: input.role, model: input.configured, reason: "missing-capability", missing, required: input.capabilities, choices });
    }
    if (isDeprecated(model, now)) {
      throw new ModelSelectionError({ role: input.role, model: input.configured, reason: "deprecated", required: input.capabilities, choices, ...(model.deprecationDate ? { deprecationDate: model.deprecationDate } : {}) });
    }
    return model.id;
  }
  for (const id of input.preferred ?? []) {
    if (choices.includes(id)) return id;
  }
  const first = choices[0];
  if (!first) throw new ModelSelectionError({ role: input.role, reason: "none-capable", required: input.capabilities, choices });
  return first;
}

// ---------------------------------------------------------------------------
// Parsing

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

function num(v: unknown): number | undefined {
  if (v === null || v === undefined || v === "") return undefined;
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : undefined;
}

function strings(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string" && x.trim() !== "").map((x) => x.trim()) : [];
}

/** Parse one entry. Returns null when it has no id. */
export function parseModelEntry(raw: unknown): AiandModel | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const m = raw as Record<string, unknown>;
  const id = str(m.id);
  if (!id) return null;
  const out: AiandModel = {
    id,
    capabilities: strings(m.capabilities).map((c) => c.toLowerCase()),
    reasoningEfforts: strings(m.reasoning_efforts),
  };
  const set = <K extends keyof AiandModel>(key: K, value: AiandModel[K] | undefined) => {
    if (value !== undefined) out[key] = value;
  };
  set("name", str(m.name) ?? str(m.display_name));
  set("provider", str(m.provider));
  set("description", str(m.description));
  set("contextWindow", num(m.context_window));
  set("maxOutputTokens", num(m.max_output_tokens));
  set("reasoningEffortDefault", str(m.reasoning_effort_default));
  set("inputPer1m", num(m.input_per_1m));
  set("outputPer1m", num(m.output_per_1m));
  set("cachedInputPer1m", num(m.cached_input_per_1m));
  set("currency", str(m.currency));
  set("deprecationDate", str(m.deprecation_date));
  return out;
}

/** Parse a `/models` body (`{ data: [...] }` or a bare array). Duplicate ids keep the first. */
export function parseModelList(body: unknown): AiandModel[] {
  const items = Array.isArray(body) ? body : body && typeof body === "object" ? (body as { data?: unknown }).data : undefined;
  if (!Array.isArray(items)) return [];
  const seen = new Set<string>();
  const out: AiandModel[] = [];
  for (const item of items) {
    const model = parseModelEntry(item);
    if (!model || seen.has(model.id)) continue;
    seen.add(model.id);
    out.push(model);
  }
  return out;
}
