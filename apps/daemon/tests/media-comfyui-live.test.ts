// Live: `comfyui-agent` through generateMedia against a real ComfyUI server
// and the ai& API. Skipped unless COMFYUI_URL and AIAND_API_KEY are set.
// COMFY_AGENT_LIVE_OUT keeps the project folder (image, trace, round images).

import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { alphaStats, decodePng } from '@open-design/comfy-agent';
import { describe, expect, it } from 'vitest';

import { generateMedia } from '../src/media/index.js';

const live = Boolean(process.env.COMFYUI_URL && process.env.AIAND_API_KEY);

describe.skipIf(!live)('live: comfyui-agent through generateMedia', () => {
  it(
    'generates a small transparent asset with a trace beside it',
    async () => {
      const keep = process.env.COMFY_AGENT_LIVE_OUT;
      const root = keep ? path.resolve(keep) : await mkdtemp(path.join(tmpdir(), 'od-comfyui-live-'));
      const projectRoot = path.join(root, 'project-root');
      const projectsRoot = path.join(projectRoot, '.od', 'projects');
      await mkdir(path.join(projectRoot, '.od'), { recursive: true });
      const savedConfigDir = process.env.OD_MEDIA_CONFIG_DIR;
      delete process.env.OD_MEDIA_CONFIG_DIR;
      delete process.env.OD_DATA_DIR;
      await writeFile(
        path.join(projectRoot, '.od', 'media-config.json'),
        JSON.stringify({ providers: { comfyui: { baseUrl: process.env.COMFYUI_URL } } }),
      );
      try {
        const started = Date.now();
        const result = await generateMedia({
          projectRoot,
          projectsRoot,
          projectId: 'live',
          surface: 'image',
          model: 'comfyui-agent',
          prompt: 'a single red paper lantern icon',
          output: 'lantern.png',
          comfy: { width: 512, height: 512, transparent: true, maxRounds: 2, negative: 'petals, flowers, text' },
          onProgress: (m) => process.stderr.write(`[live] ${m}\n`),
        });
        const dir = path.join(projectsRoot, 'live');
        const trace = JSON.parse(await readFile(path.join(dir, 'lantern.trace.json'), 'utf8'));
        const img = decodePng(await readFile(path.join(dir, 'lantern.png')));
        process.stderr.write(`[live] ${JSON.stringify({
          dir,
          elapsedMs: Date.now() - started,
          providerNote: result.providerNote,
          stopReason: trace.stopReason,
          totals: trace.totals,
          rounds: trace.rounds.map((r: { round: number; score: unknown; checks: unknown; keyColor: unknown; critic?: { items: unknown }; repair?: { applied: unknown; rejected: unknown }; timing: unknown }) => ({
            round: r.round, score: r.score, checks: r.checks, keyColor: r.keyColor, critic: r.critic?.items, applied: r.repair?.applied, rejected: r.repair?.rejected, timing: r.timing,
          })),
          checklist: trace.plan.checklist,
          alpha: alphaStats(img),
        }, null, 2)}\n`);
        expect(result.providerId).toBe('comfyui');
        expect([img.width, img.height]).toEqual([512, 512]);
        expect(trace.rounds.length).toBeGreaterThanOrEqual(1);
        expect(trace.rounds.length).toBeLessThanOrEqual(2);
        expect(trace.rounds[0].checks.find((c: { id: string }) => c.id === 'size').pass).toBe(true);
      } finally {
        if (savedConfigDir !== undefined) process.env.OD_MEDIA_CONFIG_DIR = savedConfigDir;
        if (!keep) await rm(root, { recursive: true, force: true });
      }
    },
    15 * 60 * 1000,
  );
});
