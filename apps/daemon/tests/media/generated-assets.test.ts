import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  GeneratedAssetNotFoundError,
  listGeneratingAssets,
  listGeneratedAssets,
  loadSetReferences,
  selectSetReferences,
  setGeneratedAssetReview,
} from '../../src/media/generated-assets.js';
import { assetLoopRunning, trackAssetLoop } from '../../src/media/asset-loops.js';

let dir = '';
beforeEach(async () => {
  dir = await mkdtemp(path.join(tmpdir(), 'od-generated-assets-'));
});
afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

/** Write an asset and the trace renderComfyUI writes beside it. */
async function asset(name: string, opts: { kind?: 'icon'; set?: string; finishedAt?: string; file?: boolean; workflow?: boolean } = {}) {
  const stem = name.replace(/\.png$/, '');
  await mkdir(path.dirname(path.join(dir, name)), { recursive: true });
  if (opts.file !== false) await writeFile(path.join(dir, name), Buffer.from('png'));
  const trace = opts.workflow
    ? { kind: 'comfyui-workflow', server: 'http://127.0.0.1:8188', graph: {} }
    : {
        id: `t-${stem}`,
        finishedAt: opts.finishedAt ?? '2026-10-29T00:00:00.000Z',
        request: { prompt: `a ${stem} icon`, width: 512, height: 512, ...(opts.kind ? { kind: opts.kind } : {}) },
        asset: name,
        ...(opts.set ? { set: opts.set } : {}),
      };
  await writeFile(path.join(dir, `${stem}.trace.json`), JSON.stringify(trace));
}

describe('generated assets', () => {
  it('lists every ComfyUI asset with its trace, kind, set and review state', async () => {
    await asset('icons/leaf.png', { kind: 'icon', set: 'onboarding' });
    await asset('hero.png', { workflow: true });
    await mkdir(path.join(dir, '.comfy-agent', 'leaf'), { recursive: true });
    await writeFile(path.join(dir, '.comfy-agent', 'leaf', 'trace.json'), '{}');
    const assets = await listGeneratedAssets(dir);
    expect(assets.map((a) => [a.path, a.model, a.kind, a.set, a.exists, a.review.state, a.review.reason])).toEqual([
      ['hero.png', 'comfyui-workflow', undefined, undefined, true, 'accepted', 'default'],
      ['icons/leaf.png', 'comfyui-agent', 'icon', 'onboarding', true, 'accepted', 'default'],
    ]);
    expect(assets[1]!.tracePath).toBe('icons/leaf.trace.json');
    expect(assets[1]!.prompt).toBe('a icons/leaf icon');
  });

  it('counts a deleted image as rejected', async () => {
    await asset('icons/gone.png', { kind: 'icon', file: false });
    const [gone] = await listGeneratedAssets(dir);
    expect(gone).toMatchObject({ exists: false, review: { state: 'rejected', reason: 'deleted' } });
  });

  it('stores a review in the trace file and reads it back', async () => {
    await asset('icons/leaf.png', { kind: 'icon' });
    const updated = await setGeneratedAssetReview(dir, 'icons/leaf.png', 'rejected', 'cli');
    expect(updated.review).toMatchObject({ state: 'rejected', reason: 'user', via: 'cli' });
    const onDisk = JSON.parse(await readFile(path.join(dir, 'icons/leaf.trace.json'), 'utf8'));
    expect(onDisk.review).toMatchObject({ state: 'rejected', via: 'cli' });
    expect(onDisk.request.prompt).toBe('a icons/leaf icon');
    const accepted = await setGeneratedAssetReview(dir, 'icons/leaf.png', 'accepted', 'ui');
    expect(accepted.review).toMatchObject({ state: 'accepted', reason: 'user', via: 'ui' });
  });

  it('refuses paths outside the project and assets without a trace', async () => {
    await expect(setGeneratedAssetReview(dir, '../escape.png', 'rejected', 'api')).rejects.toBeInstanceOf(GeneratedAssetNotFoundError);
    await writeFile(path.join(dir, 'plain.png'), 'x');
    await expect(setGeneratedAssetReview(dir, 'plain.png', 'rejected', 'api')).rejects.toBeInstanceOf(GeneratedAssetNotFoundError);
  });

  it('picks accepted icons for a set: same set first, newest first, never rejected, deleted or the new output', async () => {
    await asset('icons/a.png', { kind: 'icon', set: 'onboarding', finishedAt: '2026-10-29T01:00:00.000Z' });
    await asset('icons/b.png', { kind: 'icon', finishedAt: '2026-10-29T05:00:00.000Z' });
    await asset('icons/c.png', { kind: 'icon', set: 'onboarding', finishedAt: '2026-10-29T03:00:00.000Z' });
    await asset('icons/d.png', { kind: 'icon', set: 'onboarding', finishedAt: '2026-10-29T04:00:00.000Z' });
    await asset('icons/e.png', { kind: 'icon', set: 'onboarding', file: false });
    await asset('hero.png', { finishedAt: '2026-10-29T06:00:00.000Z' });
    await setGeneratedAssetReview(dir, 'icons/d.png', 'rejected', 'ui');
    const assets = await listGeneratedAssets(dir);
    expect(selectSetReferences(assets, { set: 'onboarding', exclude: 'icons/new.png' }, 3).map((a) => a.path)).toEqual([
      'icons/c.png',
      'icons/a.png',
      'icons/b.png',
    ]);
    expect(selectSetReferences(assets, { exclude: 'icons/b.png' }, 2).map((a) => a.path)).toEqual(['icons/c.png', 'icons/a.png']);
  });

  it('loads the chosen icons as set references for the asset loop', async () => {
    await asset('icons/a.png', { kind: 'icon', set: 'onboarding' });
    await asset('icons/b.png', { kind: 'icon', set: 'onboarding' });
    await setGeneratedAssetReview(dir, 'icons/b.png', 'rejected', 'ui');
    const refs = await loadSetReferences(dir, { set: 'onboarding', exclude: 'icons/new.png' }, 3);
    expect(refs).toEqual([{ id: 'icons/a.png', label: 'a icons/a icon', png: Buffer.from('png') }]);
  });
});

describe('assets being generated', () => {
  it('lists asset loops whose trace is still marked running, and not finished ones', async () => {
    await mkdir(path.join(dir, '.comfy-agent', 'hero'), { recursive: true });
    await writeFile(
      path.join(dir, '.comfy-agent', 'hero', 'trace.json'),
      JSON.stringify({ running: true, startedAt: '2026-10-29T01:00:00.000Z', request: { prompt: 'a hero photo', kind: 'icon' }, set: 'food', rounds: [{}] }),
    );
    await mkdir(path.join(dir, '.comfy-agent', 'done'), { recursive: true });
    await writeFile(path.join(dir, '.comfy-agent', 'done', 'trace.json'), JSON.stringify({ rounds: [{}] }));
    expect(await listGeneratingAssets(dir)).toEqual([
      { workDir: '.comfy-agent/hero', tracePath: '.comfy-agent/hero/trace.json', prompt: 'a hero photo', kind: 'icon', set: 'food', startedAt: '2026-10-29T01:00:00.000Z' },
    ]);
  });

  it('skips a trace still marked running when no loop in this daemon is running it (a loop cut off by a restart)', async () => {
    await mkdir(path.join(dir, '.comfy-agent', 'live'), { recursive: true });
    await writeFile(path.join(dir, '.comfy-agent', 'live', 'trace.json'), JSON.stringify({ running: true, rounds: [{}] }));
    await mkdir(path.join(dir, '.comfy-agent', 'stale'), { recursive: true });
    await writeFile(path.join(dir, '.comfy-agent', 'stale', 'trace.json'), JSON.stringify({ running: true, rounds: [{}] }));
    const live = path.join(dir, '.comfy-agent', 'live');
    const listed = await listGeneratingAssets(dir, (workDir) => workDir === live);
    expect(listed.map((a) => a.workDir)).toEqual(['.comfy-agent/live']);
  });

  it('knows which asset loops this daemon is running', async () => {
    let finish!: () => void;
    const loop = trackAssetLoop('/p/.comfy-agent/hero', new Promise<void>((r) => (finish = r)));
    expect(assetLoopRunning('/p/.comfy-agent/hero')).toBe(true);
    expect(assetLoopRunning('/p/.comfy-agent/other')).toBe(false);
    finish();
    await loop;
    expect(assetLoopRunning('/p/.comfy-agent/hero')).toBe(false);
    await trackAssetLoop('/p/.comfy-agent/bad', Promise.reject(new Error('boom'))).catch(() => undefined);
    expect(assetLoopRunning('/p/.comfy-agent/bad')).toBe(false);
  });

  it('is empty when nothing is being generated', async () => {
    expect(await listGeneratingAssets(dir)).toEqual([]);
  });
});

describe('generated assets whose files moved after generation', () => {
  /** A trace as written under a flattened output name (`assets_hero.png`), then the agent moved files. */
  async function flattenedTrace(traceAt: string) {
    await mkdir(path.dirname(path.join(dir, traceAt)), { recursive: true });
    await writeFile(path.join(dir, traceAt), JSON.stringify({ id: 't', request: { prompt: 'melon-pan', width: 1024, height: 576 }, asset: 'assets_hero.png' }));
  }

  it('finds the image at the unflattened path when only the image moved', async () => {
    await flattenedTrace('assets_hero.trace.json');
    await mkdir(path.join(dir, 'assets'), { recursive: true });
    await writeFile(path.join(dir, 'assets', 'hero.png'), Buffer.from('png'));
    const [a] = await listGeneratedAssets(dir);
    expect(a).toMatchObject({ path: 'assets/hero.png', exists: true, review: { state: 'accepted', reason: 'default' } });
  });

  it('finds the image next to the trace when both moved', async () => {
    await flattenedTrace('assets/hero.trace.json');
    await writeFile(path.join(dir, 'assets', 'hero.png'), Buffer.from('png'));
    const [a] = await listGeneratedAssets(dir);
    expect(a).toMatchObject({ path: 'assets/hero.png', exists: true, review: { state: 'accepted' } });
  });

  it('still counts a deleted image as rejected', async () => {
    await flattenedTrace('assets_hero.trace.json');
    const [a] = await listGeneratedAssets(dir);
    expect(a).toMatchObject({ path: 'assets_hero.png', exists: false, review: { state: 'rejected', reason: 'deleted' } });
  });

  it('reviews the moved image by its real path', async () => {
    await flattenedTrace('assets_hero.trace.json');
    await mkdir(path.join(dir, 'assets'), { recursive: true });
    await writeFile(path.join(dir, 'assets', 'hero.png'), Buffer.from('png'));
    const a = await setGeneratedAssetReview(dir, 'assets/hero.png', 'rejected', 'api');
    expect(a).toMatchObject({ path: 'assets/hero.png', review: { state: 'rejected', reason: 'user' } });
  });
});
