import { describe, expect, it } from "vitest";

import { LlmConfigError, LlmFormatError, LlmHttpError } from "../src/errors.js";
import { AiandClient, ANTHROPIC_VERSION, parseJsonLoose } from "../src/llm.js";
import { parseModelSettings, resolveComfyAgentConfig } from "../src/config.js";

type Captured = { url: string; headers: Record<string, string>; body: Record<string, unknown> | null };

const MODELS = {
  object: "list",
  data: [
    { id: "test/planner", capabilities: ["chat", "vision", "tool_calling"], input_per_1m: "2.000000", output_per_1m: "10.000000", cached_input_per_1m: "0.500000", currency: "usd" },
    { id: "test/critic", capabilities: ["vision"], input_per_1m: "0.100000", output_per_1m: "0.400000" },
  ],
};

function stubFetch(respond: (url: string, body: Record<string, unknown> | null) => { status?: number; json: unknown }) {
  const calls: Captured[] = [];
  const fn = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : null;
    calls.push({ url, headers: (init?.headers ?? {}) as Record<string, string>, body });
    if (url.endsWith("/models")) return new Response(JSON.stringify(MODELS), { status: 200 });
    const r = respond(url, body);
    return new Response(JSON.stringify(r.json), { status: r.status ?? 200 });
  }) as typeof fetch;
  return { fn, calls };
}

const REQ = {
  model: "test/planner",
  system: "be terse",
  content: [
    { type: "image" as const, mime: "image/png", base64: "iVBORw0KGgo=" },
    { type: "text" as const, text: "grade it" },
  ],
  schema: { type: "object", properties: { ok: { type: "boolean" } }, required: ["ok"] },
  toolName: "submit",
  reasoningEffort: "low",
};

describe("AiandClient — OpenAI shape", () => {
  it("sends chat/completions with image_url data URLs and a forced function call", async () => {
    const { fn, calls } = stubFetch(() => ({
      json: {
        model: "test/planner",
        choices: [{ message: { content: "", tool_calls: [{ type: "function", function: { name: "submit", arguments: "{\"ok\": true}" } }] } }],
        usage: { prompt_tokens: 1000, completion_tokens: 200, prompt_tokens_details: { cached_tokens: 400 }, completion_tokens_details: { reasoning_tokens: 50 } },
      },
    }));
    const client = new AiandClient({ apiKey: "k-test", baseUrl: "https://llm.example/v1", shape: "openai", fetch: fn });
    const res = await client.structured<{ ok: boolean }>(REQ);
    const call = calls.find((c) => c.url.endsWith("/chat/completions"))!;
    expect(call.headers.authorization).toBe("Bearer k-test");
    const msgs = call.body!.messages as { role: string; content: unknown }[];
    expect(msgs[0]).toEqual({ role: "system", content: "be terse" });
    expect((msgs[1]!.content as Record<string, unknown>[])[0]).toEqual({ type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } });
    expect(call.body!.tool_choice).toEqual({ type: "function", function: { name: "submit" } });
    expect(call.body!.reasoning_effort).toBe("low");
    expect(res.data).toEqual({ ok: true });
    expect(res.shape).toBe("openai");
    expect(res.usage).toEqual({ inputTokens: 1000, outputTokens: 200, cachedInputTokens: 400, reasoningTokens: 50 });
    // (600 × 2 + 400 × 0.5 + 200 × 10) / 1e6
    expect(res.estimatedCostUsd).toBeCloseTo(0.0034, 10);
    expect(res.reportedCostUsd).toBeUndefined();
  });

  it("records cost when the response reports it", async () => {
    const { fn } = stubFetch(() => ({
      json: { choices: [{ message: { tool_calls: [{ function: { name: "submit", arguments: { ok: false } } }] } }], usage: { prompt_tokens: 1, completion_tokens: 1, cost: 0.0123 } },
    }));
    const res = await new AiandClient({ apiKey: "k", fetch: fn }).structured<{ ok: boolean }>(REQ);
    expect(res.reportedCostUsd).toBe(0.0123);
    expect(res.data.ok).toBe(false);
  });

  it("falls back to JSON in the message content", async () => {
    const { fn } = stubFetch(() => ({ json: { choices: [{ message: { content: "Here:\n```json\n{\"ok\": true}\n```" } }], usage: {} } }));
    const res = await new AiandClient({ apiKey: "k", fetch: fn }).structured<{ ok: boolean }>(REQ);
    expect(res.data).toEqual({ ok: true });
  });

  it("raises LlmFormatError when there is nothing to parse and LlmHttpError on HTTP errors", async () => {
    const empty = stubFetch(() => ({ json: { choices: [{ message: { content: "" } }] } }));
    await expect(new AiandClient({ apiKey: "k", fetch: empty.fn }).structured(REQ)).rejects.toBeInstanceOf(LlmFormatError);
    const http = stubFetch(() => ({ status: 401, json: { error: "bad key" } }));
    const err = await new AiandClient({ apiKey: "k", fetch: http.fn }).structured(REQ).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(LlmHttpError);
    expect((err as LlmHttpError).status).toBe(401);
  });
});

describe("AiandClient — Anthropic shape", () => {
  it("sends messages with base64 image sources, anthropic-version and a forced tool", async () => {
    const { fn, calls } = stubFetch(() => ({
      json: {
        model: "test/critic",
        content: [{ type: "tool_use", id: "t1", name: "submit", input: { ok: true } }],
        usage: { input_tokens: 300, output_tokens: 40, cache_read_input_tokens: 100 },
      },
    }));
    const client = new AiandClient({ apiKey: "k-test", baseUrl: "https://llm.example/v1", shape: "anthropic", fetch: fn });
    const res = await client.structured<{ ok: boolean }>({ ...REQ, model: "test/critic" });
    const call = calls.find((c) => c.url.endsWith("/messages"))!;
    expect(call.headers["anthropic-version"]).toBe(ANTHROPIC_VERSION);
    expect(call.headers["x-api-key"]).toBe("k-test");
    expect(call.body!.system).toBe("be terse");
    const content = (call.body!.messages as { content: Record<string, unknown>[] }[])[0]!.content;
    expect(content[0]).toEqual({ type: "image", source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } });
    expect(call.body!.tool_choice).toEqual({ type: "tool", name: "submit" });
    expect((call.body!.tools as Record<string, unknown>[])[0]!.input_schema).toEqual(REQ.schema);
    expect(res.data).toEqual({ ok: true });
    expect(res.shape).toBe("anthropic");
    expect(res.usage).toEqual({ inputTokens: 400, outputTokens: 40, cachedInputTokens: 100 });
    expect(res.estimatedCostUsd).toBeCloseTo((400 * 0.1 + 40 * 0.4) / 1e6, 12);
  });

  it("can override the shape per call", async () => {
    const { fn, calls } = stubFetch((url) =>
      url.endsWith("/messages")
        ? { json: { content: [{ type: "tool_use", name: "submit", input: { ok: true } }], usage: {} } }
        : { json: {} },
    );
    await new AiandClient({ apiKey: "k", shape: "openai", fetch: fn }).structured(REQ, "anthropic");
    expect(calls.some((c) => c.url.endsWith("/messages"))).toBe(true);
  });
});

describe("configuration", () => {
  it("requires a key from config or AIAND_API_KEY", () => {
    const saved = process.env.AIAND_API_KEY;
    delete process.env.AIAND_API_KEY;
    try {
      expect(() => new AiandClient({})).toThrow(LlmConfigError);
      process.env.AIAND_API_KEY = "from-env";
      expect(() => new AiandClient({})).not.toThrow();
    } finally {
      if (saved === undefined) delete process.env.AIAND_API_KEY;
      else process.env.AIAND_API_KEY = saved;
    }
  });

  it("parses the compact model-settings field", () => {
    expect(parseModelSettings("planner=a/b, critic=c/d, shape=anthropic")).toEqual({ plannerModel: "a/b", criticModel: "c/d", apiShape: "anthropic" });
    expect(parseModelSettings("only/planner")).toEqual({ plannerModel: "only/planner" });
    expect(parseModelSettings("shape=messages;critic=x")).toEqual({ apiShape: "anthropic", criticModel: "x" });
    expect(parseModelSettings("")).toEqual({});
  });

  it("resolves explicit config over env over defaults", () => {
    const cfg = resolveComfyAgentConfig({ plannerModel: "explicit/p" }, { COMFYUI_URL: "http://gpu.local:8188/", COMFY_AGENT_CRITIC_MODEL: "env/c", COMFY_AGENT_API_SHAPE: "anthropic", AIAND_API_KEY: "e" });
    expect(cfg).toMatchObject({ comfyUrl: "http://gpu.local:8188", plannerModel: "explicit/p", criticModel: "env/c", apiShape: "anthropic", aiandApiKey: "e" });
    expect(resolveComfyAgentConfig({}, {})).toMatchObject({ comfyUrl: "http://127.0.0.1:8188", plannerModel: "moonshotai/kimi-k3", criticModel: "zai-org/glm-5.3-flash", apiShape: "openai" });
  });

  it("parses loose JSON", () => {
    expect(parseJsonLoose("prefix {\"a\":1} suffix", "x")).toEqual({ a: 1 });
    expect(() => parseJsonLoose("nothing", "x")).toThrow(LlmFormatError);
  });
});
