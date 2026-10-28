// Live check of the icon features against a real ComfyUI server and the ai&
// API: matting cut-out, shape references, a set reference, the near-copy
// guard, and the exact cost from the X-Cost header.
//
//   pnpm --filter @open-design/comfy-agent live:icons
//
// Runs only when COMFY_AGENT_LIVE_ICONS=1, COMFYUI_URL and AIAND_API_KEY are
// set; it queues several FLUX.2 [klein] jobs, so never alongside an eval.
// The server needs the flux2-klein-4b and matting model sets
// (scripts/comfyui-local/setup.ts). Optional: COMFY_AGENT_LIVE_OUT keeps the
// images and traces; COMFY_AGENT_REFERENCE_CACHE reuses a reference cache.

import { mkdtemp, mkdir, readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { ComfyClient } from "../src/client.js";
import { resolveComfyAgentConfig } from "../src/config.js";
import { alphaStats, decodePng } from "../src/image.js";
import { AiandClient } from "../src/llm.js";
import { generateAsset } from "../src/loop.js";
import { ReferenceLibrary } from "../src/references.js";

const live = process.env.COMFY_AGENT_LIVE_ICONS === "1" && Boolean(process.env.COMFYUI_URL && process.env.AIAND_API_KEY);

describe.skipIf(!live)("live: icon references, set references and matting", () => {
  it(
    "makes a transparent icon from a shape reference, then a second icon of the same set",
    async () => {
      const cfg = resolveComfyAgentConfig();
      const comfy = new ComfyClient({ baseUrl: cfg.comfyUrl });
      const llm = new AiandClient({ ...(cfg.aiandApiKey ? { apiKey: cfg.aiandApiKey } : {}), baseUrl: cfg.aiandBaseUrl, shape: cfg.apiShape });
      const outDir = process.env.COMFY_AGENT_LIVE_OUT ? path.resolve(process.env.COMFY_AGENT_LIVE_OUT) : await mkdtemp(path.join(os.tmpdir(), "comfy-agent-live-icons-"));
      await mkdir(outDir, { recursive: true });
      const library = new ReferenceLibrary({ cacheDir: process.env.COMFY_AGENT_REFERENCE_CACHE ?? path.join(outDir, "reference-cache") });
      const common = {
        comfy,
        llm,
        plannerModel: cfg.plannerModel,
        criticModel: cfg.criticModel,
        plannerReasoningEffort: cfg.plannerReasoningEffort ?? "low",
        criticReasoningEffort: cfg.criticReasoningEffort ?? "low",
        limits: { maxRounds: 2 },
      };

      const first = await generateAsset(
        { prompt: "A coffee cup icon with steam, glossy 3D style, violet palette.", width: 512, height: 512, transparent: true, kind: "icon", seed: 7 },
        { ...common, workDir: path.join(outDir, "coffee"), references: { library } },
      );
      const t1 = first.trace;
      expect(t1.server.cutout).toMatchObject({ method: "matte" });
      expect(t1.references!.candidates.length).toBeGreaterThan(0);
      // The planner may choose no reference; when it chose one, the near-copy guard ran on every round.
      if (t1.references!.shape.length) {
        expect(t1.references!.shape[0]!.source).toBeTruthy();
        expect(t1.rounds.every((r) => r.checks.some((c) => c.id === "reference_copy"))).toBe(true);
      }
      expect(t1.rounds[0]!.checks.map((c) => c.id)).toContain("matte_residue");
      expect(alphaStats(decodePng(first.png)).borderTransparent).toBeGreaterThan(0.9);
      // X-Aiand-Metrics: every planner and critic call reports its exact cost.
      expect(t1.totals.reportedCostUsd).not.toBeNull();
      expect(t1.plan.call?.reportedCost?.source).toBe("x-cost");

      const second = await generateAsset(
        { prompt: "A notification bell icon, glossy 3D style, violet palette.", width: 512, height: 512, transparent: true, kind: "icon", seed: 8 },
        { ...common, workDir: path.join(outDir, "bell"), references: { library, set: [{ id: "coffee.png", png: first.png, label: "coffee cup" }] } },
      );
      const t2 = second.trace;
      expect(t2.references!.set).toEqual([{ id: "coffee.png", label: "coffee cup" }]);
      expect(t2.plan.checklist.map((c) => c.id)).toContain("set_consistency");
      expect(t2.rounds[0]!.referenceSimilarity!.some((s) => s.id === "coffee.png")).toBe(true);
      const graph = t2.rounds[0]!.graph;
      expect(Object.values(graph).filter((n) => n.class_type === "ReferenceLatent").length).toBeGreaterThanOrEqual(2);
      const onDisk = JSON.parse(await readFile(path.join(outDir, "bell", "trace.json"), "utf8"));
      expect(onDisk.references.set[0].id).toBe("coffee.png");
    },
    30 * 60 * 1000,
  );
});
