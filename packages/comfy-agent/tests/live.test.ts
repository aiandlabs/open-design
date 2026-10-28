// Live end-to-end run against a real ComfyUI server and the ai& API.
// Skipped unless COMFYUI_URL and AIAND_API_KEY are both set. Optional:
//   COMFY_AGENT_LIVE_OUT   keep the output image and trace in this directory
//   COMFY_AGENT_API_SHAPE  openai | anthropic

import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { ComfyClient } from "../src/client.js";
import { resolveComfyAgentConfig } from "../src/config.js";
import { decodePng } from "../src/image.js";
import { AiandClient } from "../src/llm.js";
import { generateAsset } from "../src/loop.js";

const live = Boolean(process.env.COMFYUI_URL && process.env.AIAND_API_KEY);

describe.skipIf(!live)("live: ComfyUI + ai& asset loop", () => {
  it(
    "generates a small asset end to end",
    async () => {
      const cfg = resolveComfyAgentConfig();
      const comfy = new ComfyClient({ baseUrl: cfg.comfyUrl });
      const llm = new AiandClient({ ...(cfg.aiandApiKey ? { apiKey: cfg.aiandApiKey } : {}), baseUrl: cfg.aiandBaseUrl, shape: cfg.apiShape });
      const outDir = process.env.COMFY_AGENT_LIVE_OUT
        ? path.resolve(process.env.COMFY_AGENT_LIVE_OUT)
        : await mkdtemp(path.join(os.tmpdir(), "comfy-agent-live-"));
      await mkdir(outDir, { recursive: true });
      const res = await generateAsset(
        { prompt: "a red paper lantern on a white background", width: 768, height: 768, negative: "petals, flowers" },
        {
          comfy,
          llm,
          plannerModel: cfg.plannerModel,
          criticModel: cfg.criticModel,
          plannerReasoningEffort: cfg.plannerReasoningEffort ?? "low",
          criticReasoningEffort: cfg.criticReasoningEffort ?? "low",
          workDir: outDir,
          limits: { maxRounds: 2, maxTimeMs: 12 * 60 * 1000, maxCostUsd: 0.5 },
          runTimeoutMs: 5 * 60 * 1000,
          onProgress: (m) => process.stderr.write(`[live] ${m}\n`),
        },
      );
      await writeFile(path.join(outDir, "asset.png"), res.png);
      const summary = {
        outDir,
        stopReason: res.trace.stopReason,
        rounds: res.trace.rounds.map((r) => ({
          round: r.round,
          score: r.score,
          checks: r.checks,
          critic: r.critic?.items,
          applied: r.repair?.applied,
          rejected: r.repair?.rejected,
          timing: r.timing,
          durationMs: r.durationMs,
        })),
        checklist: res.trace.plan.checklist,
        totals: res.trace.totals,
        best: res.trace.best,
      };
      process.stderr.write(`[live] ${JSON.stringify(summary, null, 2)}\n`);
      const img = decodePng(res.png);
      expect([img.width, img.height]).toEqual([768, 768]);
      expect(res.trace.rounds.length).toBeGreaterThanOrEqual(1);
      expect(res.trace.rounds.length).toBeLessThanOrEqual(2);
      expect(res.trace.rounds[0]!.critic?.error).toBeUndefined();
    },
    15 * 60 * 1000,
  );
});
