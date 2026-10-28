// Minimal ai& API client over plain fetch. It speaks both API shapes the
// service offers, as equals:
//
//   openai    → POST {base}/chat/completions, images as `image_url` data URLs,
//               structured output via a forced function tool call.
//   anthropic → POST {base}/messages, images as base64 `image` sources,
//               structured output via a forced `tool_use`, `anthropic-version`.
//
// Both return the same StructuredResult: parsed JSON, token usage, the cost
// the response reported (if any) and a cost estimate from the per-token
// prices `GET {base}/models` publishes.

import { LlmConfigError, LlmFormatError, LlmHttpError } from "./errors.js";

export type ApiShape = "openai" | "anthropic";

export const DEFAULT_AIAND_BASE_URL = "https://api.aiand.com/v1";
export const DEFAULT_PLANNER_MODEL = "moonshotai/kimi-k3";
export const DEFAULT_CRITIC_MODEL = "zai-org/glm-5.3-flash";
export const ANTHROPIC_VERSION = "2023-06-01";

export type LlmClientOptions = {
  apiKey?: string;
  baseUrl?: string;
  shape?: ApiShape;
  fetch?: typeof fetch;
  requestTimeoutMs?: number;
};

export type ContentPart = { type: "text"; text: string } | { type: "image"; mime: string; base64: string };

export type JsonSchema = Record<string, unknown>;

export type StructuredRequest = {
  model: string;
  system: string;
  content: ContentPart[];
  /** JSON schema for the tool input / response object. */
  schema: JsonSchema;
  toolName: string;
  toolDescription?: string;
  maxTokens?: number;
  /** Passed through as `reasoning_effort` (openai shape) when set. */
  reasoningEffort?: string;
  temperature?: number;
};

export type TokenUsage = {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens?: number;
  reasoningTokens?: number;
};

export type StructuredResult<T> = {
  data: T;
  model: string;
  shape: ApiShape;
  usage: TokenUsage;
  /** Cost the API reported in the response, when it reports one. */
  reportedCostUsd?: number;
  /** Cost computed from `/models` prices and the usage above. */
  estimatedCostUsd?: number;
  latencyMs: number;
};

export type ModelPricing = {
  id: string;
  capabilities: string[];
  inputPer1m?: number;
  outputPer1m?: number;
  cachedInputPer1m?: number;
  currency?: string;
};

export function resolveApiKey(explicit?: string, env: NodeJS.ProcessEnv = process.env): string | undefined {
  const fromArg = explicit?.trim();
  if (fromArg) return fromArg;
  const fromEnv = env.AIAND_API_KEY?.trim();
  return fromEnv || undefined;
}

export class AiandClient {
  readonly baseUrl: string;
  readonly shape: ApiShape;
  private readonly apiKey: string;
  private readonly fetchImpl: typeof fetch;
  private readonly requestTimeoutMs: number;
  private modelsCache: Promise<Map<string, ModelPricing>> | null = null;

  constructor(options: LlmClientOptions = {}) {
    const apiKey = resolveApiKey(options.apiKey);
    if (!apiKey) throw new LlmConfigError("no ai& API key: pass apiKey or set AIAND_API_KEY");
    this.apiKey = apiKey;
    this.baseUrl = (options.baseUrl ?? DEFAULT_AIAND_BASE_URL).replace(/\/+$/, "");
    this.shape = options.shape ?? "openai";
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.requestTimeoutMs = options.requestTimeoutMs ?? 180_000;
  }

  private headers(shape: ApiShape): Record<string, string> {
    return shape === "anthropic"
      ? { "x-api-key": this.apiKey, authorization: `Bearer ${this.apiKey}`, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json" }
      : { authorization: `Bearer ${this.apiKey}`, "content-type": "application/json" };
  }

  /** `GET /models`, cached. Includes per-token prices used for cost estimates. */
  listModels(): Promise<Map<string, ModelPricing>> {
    if (!this.modelsCache) {
      const pending = (async () => {
        const url = `${this.baseUrl}/models`;
        const res = await this.fetchImpl(url, { headers: this.headers("openai"), signal: AbortSignal.timeout(this.requestTimeoutMs) });
        const text = await res.text();
        if (!res.ok) throw new LlmHttpError(url, res.status, text);
        const body = JSON.parse(text) as { data?: Record<string, unknown>[] };
        const map = new Map<string, ModelPricing>();
        for (const m of body.data ?? []) {
          const id = String(m.id ?? "");
          if (!id) continue;
          const price = (k: string) => (m[k] === undefined || m[k] === null ? undefined : Number(m[k]));
          map.set(id, {
            id,
            capabilities: Array.isArray(m.capabilities) ? m.capabilities.map(String) : [],
            ...(price("input_per_1m") === undefined ? {} : { inputPer1m: price("input_per_1m")! }),
            ...(price("output_per_1m") === undefined ? {} : { outputPer1m: price("output_per_1m")! }),
            ...(price("cached_input_per_1m") === undefined ? {} : { cachedInputPer1m: price("cached_input_per_1m")! }),
            ...(typeof m.currency === "string" ? { currency: m.currency } : {}),
          });
        }
        return map;
      })();
      this.modelsCache = pending;
      pending.catch(() => {
        if (this.modelsCache === pending) this.modelsCache = null;
      });
    }
    return this.modelsCache;
  }

  async structured<T>(req: StructuredRequest, shape: ApiShape = this.shape): Promise<StructuredResult<T>> {
    const started = Date.now();
    const { url, body } = shape === "anthropic" ? this.anthropicRequest(req) : this.openaiRequest(req);
    let res: Response;
    try {
      res = await this.fetchImpl(url, {
        method: "POST",
        headers: this.headers(shape),
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.requestTimeoutMs),
      });
    } catch (err) {
      throw new LlmHttpError(url, 0, `request failed: ${err instanceof Error ? err.message : String(err)}`);
    }
    const text = await res.text();
    if (!res.ok) throw new LlmHttpError(url, res.status, text);
    let json: Record<string, unknown>;
    try {
      json = JSON.parse(text);
    } catch {
      throw new LlmFormatError(`ai& API returned non-JSON from ${url}`, text);
    }
    const parsed = shape === "anthropic" ? parseAnthropic(json, req.toolName) : parseOpenAI(json, req.toolName);
    const usage = shape === "anthropic" ? anthropicUsage(json) : openaiUsage(json);
    const reported = reportedCost(json);
    const result: StructuredResult<T> = {
      data: parsed as T,
      model: typeof json.model === "string" ? json.model : req.model,
      shape,
      usage,
      latencyMs: Date.now() - started,
      ...(reported === undefined ? {} : { reportedCostUsd: reported }),
    };
    const estimate = await this.estimateCost(req.model, usage).catch(() => undefined);
    if (estimate !== undefined) result.estimatedCostUsd = estimate;
    return result;
  }

  async estimateCost(model: string, usage: TokenUsage): Promise<number | undefined> {
    const pricing = (await this.listModels()).get(model);
    if (!pricing || pricing.inputPer1m === undefined || pricing.outputPer1m === undefined) return undefined;
    const cached = usage.cachedInputTokens ?? 0;
    const uncached = Math.max(0, usage.inputTokens - cached);
    const cachedRate = pricing.cachedInputPer1m ?? pricing.inputPer1m;
    return (uncached * pricing.inputPer1m + cached * cachedRate + usage.outputTokens * pricing.outputPer1m) / 1_000_000;
  }

  private openaiRequest(req: StructuredRequest): { url: string; body: Record<string, unknown> } {
    const content = req.content.map((p) =>
      p.type === "text"
        ? { type: "text", text: p.text }
        : { type: "image_url", image_url: { url: `data:${p.mime};base64,${p.base64}` } },
    );
    const body: Record<string, unknown> = {
      model: req.model,
      max_tokens: req.maxTokens ?? 4096,
      messages: [
        { role: "system", content: req.system },
        { role: "user", content },
      ],
      tools: [{ type: "function", function: { name: req.toolName, description: req.toolDescription ?? "Return the result.", parameters: req.schema } }],
      tool_choice: { type: "function", function: { name: req.toolName } },
    };
    if (req.reasoningEffort) body.reasoning_effort = req.reasoningEffort;
    if (req.temperature !== undefined) body.temperature = req.temperature;
    return { url: `${this.baseUrl}/chat/completions`, body };
  }

  private anthropicRequest(req: StructuredRequest): { url: string; body: Record<string, unknown> } {
    const content = req.content.map((p) =>
      p.type === "text"
        ? { type: "text", text: p.text }
        : { type: "image", source: { type: "base64", media_type: p.mime, data: p.base64 } },
    );
    const body: Record<string, unknown> = {
      model: req.model,
      max_tokens: req.maxTokens ?? 4096,
      system: req.system,
      messages: [{ role: "user", content }],
      tools: [{ name: req.toolName, description: req.toolDescription ?? "Return the result.", input_schema: req.schema }],
      tool_choice: { type: "tool", name: req.toolName },
    };
    if (req.temperature !== undefined) body.temperature = req.temperature;
    return { url: `${this.baseUrl}/messages`, body };
  }
}

function parseOpenAI(json: Record<string, unknown>, toolName: string): unknown {
  const choices = Array.isArray(json.choices) ? json.choices : [];
  const message = (choices[0] as { message?: Record<string, unknown> } | undefined)?.message ?? {};
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const call = (calls as { function?: { name?: string; arguments?: unknown } }[]).find((c) => c.function?.name === toolName) ?? calls[0];
  const args = (call as { function?: { arguments?: unknown } } | undefined)?.function?.arguments;
  if (args !== undefined) {
    if (typeof args === "object" && args !== null) return args;
    return parseJsonLoose(String(args), "tool call arguments");
  }
  // Fall back to JSON in the text content.
  const content = typeof message.content === "string" ? message.content : "";
  if (content.trim()) return parseJsonLoose(content, "message content");
  throw new LlmFormatError(`model returned no ${toolName} tool call and no JSON content`, JSON.stringify(json));
}

function parseAnthropic(json: Record<string, unknown>, toolName: string): unknown {
  const blocks = Array.isArray(json.content) ? (json.content as Record<string, unknown>[]) : [];
  const tool = blocks.find((b) => b.type === "tool_use" && b.name === toolName) ?? blocks.find((b) => b.type === "tool_use");
  if (tool && tool.input && typeof tool.input === "object") return tool.input;
  const text = blocks.filter((b) => b.type === "text").map((b) => String(b.text ?? "")).join("\n");
  if (text.trim()) return parseJsonLoose(text, "text content");
  throw new LlmFormatError(`model returned no ${toolName} tool_use block and no JSON text`, JSON.stringify(json));
}

/** Parse JSON that may be wrapped in a code fence or surrounded by prose. */
export function parseJsonLoose(text: string, where: string): unknown {
  const attempts = [text.trim()];
  const fence = /```(?:json)?\s*([\s\S]*?)```/i.exec(text);
  if (fence?.[1]) attempts.push(fence[1].trim());
  const first = text.indexOf("{");
  const last = text.lastIndexOf("}");
  if (first >= 0 && last > first) attempts.push(text.slice(first, last + 1));
  for (const a of attempts) {
    try {
      return JSON.parse(a);
    } catch {
      // try next
    }
  }
  throw new LlmFormatError(`could not parse JSON from ${where}`, text);
}

function openaiUsage(json: Record<string, unknown>): TokenUsage {
  const u = (json.usage ?? {}) as Record<string, unknown>;
  const pd = (u.prompt_tokens_details ?? {}) as Record<string, unknown>;
  const cd = (u.completion_tokens_details ?? {}) as Record<string, unknown>;
  return {
    inputTokens: Number(u.prompt_tokens ?? 0),
    outputTokens: Number(u.completion_tokens ?? 0),
    ...(typeof pd.cached_tokens === "number" ? { cachedInputTokens: pd.cached_tokens } : {}),
    ...(typeof cd.reasoning_tokens === "number" ? { reasoningTokens: cd.reasoning_tokens } : {}),
  };
}

function anthropicUsage(json: Record<string, unknown>): TokenUsage {
  const u = (json.usage ?? {}) as Record<string, unknown>;
  const cachedRead = typeof u.cache_read_input_tokens === "number" ? u.cache_read_input_tokens : 0;
  return {
    // Anthropic reports cached reads separately from input_tokens.
    inputTokens: Number(u.input_tokens ?? 0) + cachedRead,
    outputTokens: Number(u.output_tokens ?? 0),
    ...(typeof u.cache_read_input_tokens === "number" ? { cachedInputTokens: cachedRead } : {}),
  };
}

/** Cost if the response reports one (`usage.cost`, `cost`, or `usage.total_cost`). */
function reportedCost(json: Record<string, unknown>): number | undefined {
  const u = (json.usage ?? {}) as Record<string, unknown>;
  for (const v of [u.cost, json.cost, u.total_cost, (u.cost_details as Record<string, unknown> | undefined)?.total]) {
    const n = typeof v === "string" ? Number(v) : v;
    if (typeof n === "number" && Number.isFinite(n)) return n;
  }
  return undefined;
}
