import { describe, expect, it } from "vitest";

import { LlmHttpError, ModelSelectionError } from "../src/errors.js";
import { AiandClient } from "../src/llm.js";
import {
  AiandModelCatalog,
  ROLE_CAPABILITIES,
  parseModelEntry,
  parseModelList,
  selectModel,
  sharedModelCatalog,
} from "../src/models.js";

const NOW = Date.parse("2026-10-28T00:00:00Z");

// Shaped like the live `GET https://api.aiand.com/v1/models` response.
function entry(id: string, capabilities: string[], extra: Record<string, unknown> = {}) {
  return {
    id,
    name: id.toUpperCase(),
    object: "model",
    provider: id.split("/")[0],
    capabilities,
    context_window: 262144,
    max_output_tokens: 32768,
    reasoning_efforts: ["none", "high"],
    reasoning_effort_default: "high",
    description: `${id} test entry`,
    currency: "usd",
    input_per_1m: "0.300000",
    output_per_1m: "1.200000",
    cached_input_per_1m: "0.030000",
    deprecation_date: null,
    ...extra,
  };
}

const VISION_TOOLS = ["reasoning", "tool_calling", "vision", "document"];
const TEXT_TOOLS = ["chat", "reasoning", "tool_calling"];

function stubCatalog(data: unknown[], opts: { status?: number; now?: () => number; ttlMs?: number } = {}) {
  const calls: { url: string; headers: Record<string, string> }[] = [];
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(input), headers: (init?.headers ?? {}) as Record<string, string> });
    return new Response(JSON.stringify({ object: "list", data }), { status: opts.status ?? 200 });
  }) as typeof fetch;
  const catalog = new AiandModelCatalog({
    baseUrl: "https://llm.test/v1/",
    fetch: fetchImpl,
    now: opts.now ?? (() => NOW),
    ...(opts.ttlMs !== undefined ? { ttlMs: opts.ttlMs } : {}),
  });
  return { catalog, calls };
}

describe("parsing /v1/models", () => {
  it("reads every documented field and converts string prices", () => {
    const m = parseModelEntry(entry("moonshotai/kimi-k3", VISION_TOOLS, { deprecation_date: "2027-01-31" }))!;
    expect(m).toEqual({
      id: "moonshotai/kimi-k3",
      name: "MOONSHOTAI/KIMI-K3",
      provider: "moonshotai",
      description: "moonshotai/kimi-k3 test entry",
      capabilities: VISION_TOOLS,
      contextWindow: 262144,
      maxOutputTokens: 32768,
      reasoningEfforts: ["none", "high"],
      reasoningEffortDefault: "high",
      inputPer1m: 0.3,
      outputPer1m: 1.2,
      cachedInputPer1m: 0.03,
      currency: "usd",
      deprecationDate: "2027-01-31",
    });
  });

  it("skips entries without an id, tolerates junk fields and keeps the first duplicate", () => {
    const list = parseModelList({
      data: [
        null,
        "nope",
        { name: "no id" },
        { id: "a/x", capabilities: "vision", input_per_1m: "free", reasoning_efforts: [1, "low"] },
        { id: "a/x", capabilities: ["vision"] },
        { id: "b/y", capabilities: ["Vision", 3, "TOOL_CALLING"] },
      ],
    });
    expect(list.map((m) => m.id)).toEqual(["a/x", "b/y"]);
    expect(list[0]).toEqual({ id: "a/x", capabilities: [], reasoningEfforts: ["low"] });
    expect(list[1]!.capabilities).toEqual(["vision", "tool_calling"]);
    expect(parseModelList({ error: "x" })).toEqual([]);
    expect(parseModelList([{ id: "bare/array" }]).map((m) => m.id)).toEqual(["bare/array"]);
  });
});

describe("default selection", () => {
  const live = [
    entry("deepseek-ai/deepseek-v4-pro", ["reasoning", "tool_calling"]),
    entry("qwen/qwen3.8-27b", [...VISION_TOOLS, "video"]),
    entry("zai-org/glm-5.3-flash", [...VISION_TOOLS, "chat"]),
    entry("moonshotai/kimi-k3", VISION_TOOLS),
  ];

  it("uses the preferred id when the live list offers it", async () => {
    const { catalog } = stubCatalog(live);
    expect(await catalog.resolve("planner")).toBe("moonshotai/kimi-k3");
    expect(await catalog.resolve("critic")).toBe("zai-org/glm-5.3-flash");
    expect(await catalog.resolve("visual-check")).toBe("zai-org/glm-5.3-flash");
  });

  it("falls back to the first capable model when no preferred id is listed", async () => {
    const { catalog } = stubCatalog(live.filter((m) => m.id !== "moonshotai/kimi-k3" && m.id !== "zai-org/glm-5.3-flash"));
    expect(await catalog.resolve("planner")).toBe("qwen/qwen3.8-27b");
    expect(await catalog.resolve("critic")).toBe("qwen/qwen3.8-27b");
  });

  it("never uses a preferred id that the live list lacks the capabilities for", async () => {
    const { catalog } = stubCatalog([
      entry("moonshotai/kimi-k3", ["reasoning", "tool_calling"]),
      entry("google/gemma-4-31b-it", VISION_TOOLS),
    ]);
    expect(await catalog.resolve("planner")).toBe("google/gemma-4-31b-it");
  });

  it("ignores models whose deprecation date has passed, but not future ones", async () => {
    const { catalog } = stubCatalog([
      entry("moonshotai/kimi-k3", VISION_TOOLS, { deprecation_date: "2026-10-01" }),
      entry("old/vision", VISION_TOOLS, { deprecation_date: "2026-10-27T23:59:59Z" }),
      entry("next/vision", VISION_TOOLS, { deprecation_date: "2027-03-01" }),
    ]);
    expect(await catalog.resolve("planner")).toBe("next/vision");
  });

  it("raises a typed error when no model has the required capabilities", async () => {
    const { catalog } = stubCatalog([entry("text/only", TEXT_TOOLS), entry("vision/no-tools", ["vision", "chat"])]);
    const err = await catalog.resolve("critic").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ModelSelectionError);
    expect(err).toMatchObject({ code: "LLM_MODEL", role: "critic", reason: "none-capable", choices: [], required: ["vision", "tool_calling"] });
    expect((err as Error).message).toBe("no ai& model has vision, tool_calling for the critic; the ai& catalog lists no valid model");
  });

  it("requires vision and tool calling for every role", () => {
    expect(ROLE_CAPABILITIES).toEqual({
      planner: ["vision", "tool_calling"],
      critic: ["vision", "tool_calling"],
      "visual-check": ["vision", "tool_calling"],
    });
  });
});

describe("configured models", () => {
  const live = [
    entry("text/only", TEXT_TOOLS),
    entry("zai-org/glm-5.3-flash", VISION_TOOLS),
    entry("qwen/qwen3.8-27b", VISION_TOOLS),
    entry("gone/vision", VISION_TOOLS, { deprecation_date: "2026-09-30" }),
  ];

  it("accepts a configured id that exists with the capabilities, over the preference", async () => {
    const { catalog } = stubCatalog(live);
    expect(await catalog.resolve("critic", "  qwen/qwen3.8-27b ")).toBe("qwen/qwen3.8-27b");
  });

  it("treats a blank configured id as unset", async () => {
    const { catalog } = stubCatalog(live);
    expect(await catalog.resolve("critic", "   ")).toBe("zai-org/glm-5.3-flash");
    expect(await catalog.resolve("critic", null)).toBe("zai-org/glm-5.3-flash");
  });

  it("rejects an unknown id and lists the valid choices", async () => {
    const { catalog } = stubCatalog(live);
    const err = (await catalog.resolve("planner", "nope/missing").catch((e: unknown) => e)) as ModelSelectionError;
    expect(err).toBeInstanceOf(ModelSelectionError);
    expect(err.reason).toBe("unknown");
    expect(err.choices).toEqual(["zai-org/glm-5.3-flash", "qwen/qwen3.8-27b"]);
    expect(err.message).toBe('planner model "nope/missing" is not in the ai& model list; valid planner models: zai-org/glm-5.3-flash, qwen/qwen3.8-27b');
  });

  it("rejects an id without vision and names the missing capability", async () => {
    const { catalog } = stubCatalog(live);
    const err = (await catalog.resolve("visual-check", "text/only").catch((e: unknown) => e)) as ModelSelectionError;
    expect(err).toBeInstanceOf(ModelSelectionError);
    expect(err).toMatchObject({ reason: "missing-capability", missing: ["vision"], model: "text/only" });
    expect(err.message).toContain('visual-check model "text/only" lacks vision (the visual-check needs vision, tool_calling)');
    expect(err.message).toContain("valid visual-check models: zai-org/glm-5.3-flash, qwen/qwen3.8-27b");
  });

  it("rejects a deprecated id", async () => {
    const { catalog } = stubCatalog(live);
    await expect(catalog.resolve("critic", "gone/vision")).rejects.toMatchObject({ reason: "deprecated", deprecationDate: "2026-09-30" });
  });

  it("honours custom requirements", () => {
    const models = parseModelList({ data: live });
    expect(selectModel(models, { role: "planner", capabilities: ["tool_calling"], preferred: ["text/only"], now: NOW })).toBe("text/only");
  });
});

describe("catalog fetching and caching", () => {
  it("fetches once per TTL, without a key when none is given", async () => {
    let now = NOW;
    const { catalog, calls } = stubCatalog([entry("a/v", VISION_TOOLS)], { now: () => now, ttlMs: 1000 });
    await catalog.list();
    await catalog.resolve("critic");
    await catalog.get("a/v");
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe("https://llm.test/v1/models");
    expect(calls[0]!.headers).toEqual({});
    now += 1001;
    await catalog.list();
    expect(calls).toHaveLength(2);
  });

  it("does not cache a failure", async () => {
    let fail = true;
    const fetchImpl = (async () =>
      fail ? new Response("down", { status: 503 }) : new Response(JSON.stringify({ data: [entry("a/v", VISION_TOOLS)] }))) as typeof fetch;
    const catalog = new AiandModelCatalog({ baseUrl: "https://llm.test/v1", fetch: fetchImpl });
    await expect(catalog.list()).rejects.toBeInstanceOf(LlmHttpError);
    fail = false;
    expect((await catalog.list()).map((m) => m.id)).toEqual(["a/v"]);
  });

  it("sends the key when one is given", async () => {
    const seen: Record<string, string>[] = [];
    const fetchImpl = (async (_: unknown, init?: RequestInit) => {
      seen.push((init?.headers ?? {}) as Record<string, string>);
      return new Response(JSON.stringify({ data: [] }));
    }) as typeof fetch;
    await new AiandModelCatalog({ apiKey: "k-test", fetch: fetchImpl }).list();
    expect(seen[0]).toEqual({ authorization: "Bearer k-test" });
  });

  it("shares one catalog per base URL and key", () => {
    const a = sharedModelCatalog({ baseUrl: "https://x.test/v1/" });
    expect(sharedModelCatalog({ baseUrl: "https://x.test/v1" })).toBe(a);
    expect(sharedModelCatalog({ baseUrl: "https://x.test/v1", apiKey: "k" })).not.toBe(a);
  });

  it("is what AiandClient uses for cost estimates and model resolution", async () => {
    const { catalog, calls } = stubCatalog([entry("zai-org/glm-5.3-flash", VISION_TOOLS)]);
    const client = new AiandClient({ apiKey: "k", catalog });
    expect(client.catalog).toBe(catalog);
    expect(await client.resolveModel("critic")).toBe("zai-org/glm-5.3-flash");
    // (1000 × 0.3 + 500 × 1.2) / 1e6
    expect(await client.estimateCost("zai-org/glm-5.3-flash", { inputTokens: 1000, outputTokens: 500 })).toBeCloseTo(0.0009, 12);
    expect(calls).toHaveLength(1);
  });
});
