import { mkdtemp, readFile, rm } from 'node:fs/promises';
import type { Server } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';

import express from 'express';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { VisualCheckResponse } from '@open-design/contracts';
import { AiandModelCatalog } from '@open-design/comfy-agent';

import { registerVisualCheckRoutes, type RegisterVisualCheckRoutesDeps } from '../src/routes/visual-check.js';
import { VISUAL_CHECK_BASELINE_CHECKLIST } from '../src/visual-check/grade.js';
import { VisualCheckRenderError } from '../src/visual-check/render.js';
import { readProjectOriginalPrompt } from '../src/visual-check/route-deps.js';

// A real 2x3 PNG header is enough for the dimension probe.
function fakePng(width: number, height: number): Buffer {
  const buf = Buffer.alloc(33);
  buf.writeUInt32BE(0x89504e47, 0);
  buf.writeUInt32BE(0x0d0a1a0a, 4);
  buf.writeUInt32BE(13, 8);
  buf.write('IHDR', 12, 'ascii');
  buf.writeUInt32BE(width, 16);
  buf.writeUInt32BE(height, 20);
  return buf;
}

let server: Server | undefined;
let baseUrl = '';
let projectDir = '';

type Setup = {
  key?: string | null;
  rounds?: number;
  modelResponse?: unknown;
  renderError?: Error;
  projectPrompt?: string | null;
  grant?: { projectId: string; runId: string } | null;
  /** Settings model; undefined keeps 'settings/model', null clears it. */
  prefsModel?: string | null;
};

// A stubbed `GET /v1/models`, shaped like the live ai& list.
const CATALOG = [
  { id: 'text/only', capabilities: ['chat', 'tool_calling'] },
  { id: 'catalog/vision', capabilities: ['vision', 'tool_calling'] },
  { id: 'settings/model', capabilities: ['vision', 'tool_calling', 'reasoning'] },
  { id: 'other/model', capabilities: ['vision', 'tool_calling'] },
];

async function start(setup: Setup = {}) {
  const structured = vi.fn(async (req: any) => ({
    data: setup.modelResponse ?? {
      summary: 'Heading clipped on the right.',
      items: [
        { id: 'b1', item: 'x', pass: false, evidence: "The hero heading 'Quarterly' runs past the right edge.", repairHint: 'Allow wrapping.', viewport: 'desktop' },
        { id: 'b2', item: 'x', pass: true, evidence: 'Body text is dark on white.', repairHint: 'ignored for passes' },
        { id: 'b3', item: 'x', pass: true, evidence: 'ok', repairHint: '' },
        { id: 'b4', item: 'x', pass: true, evidence: 'ok', repairHint: '' },
        { id: 'b5', item: 'x', pass: true, evidence: 'ok', repairHint: '' },
        { id: 'd1', item: 'Three pricing tiers are visible', pass: true, evidence: 'Three cards.', repairHint: '' },
        { id: 'zz', item: 'not ours', pass: true, evidence: '', repairHint: '' },
      ],
    },
    model: req.model,
    shape: 'openai' as const,
    usage: { inputTokens: 1200, outputTokens: 300 },
    latencyMs: 5,
    estimatedCostUsd: 0.0012,
  }));
  const catalog = new AiandModelCatalog({
    baseUrl: 'https://llm.test/v1',
    fetch: (async () => new Response(JSON.stringify({ data: CATALOG }))) as typeof fetch,
  });
  const resolveModel = (role: 'visual-check', configured?: string | null) => catalog.resolve(role, configured);
  const render = vi.fn(async ({ viewports }: { viewports: Array<'desktop' | 'mobile'> }) => {
    if (setup.renderError) throw setup.renderError;
    return {
      renderer: 'system-browser' as const,
      images: viewports.map((viewport) => ({
        viewport,
        png: viewport === 'desktop' ? fakePng(1440, 1200) : fakePng(390, 1800),
        width: viewport === 'desktop' ? 1440 : 390,
        height: viewport === 'desktop' ? 1200 : 1800,
      })),
    };
  });
  const deps: RegisterVisualCheckRoutesDeps = {
    db: {},
    getProject: (_db, id) => (id === 'p1' ? { id: 'p1', metadata: {} } : null),
    sendApiError: (res, status, code, message) => res.status(status).json({ error: { code, message } }),
    isLocalSameOrigin: () => true,
    resolvedPortRef: { current: 0 },
    authorizeProjectRequest: (async () => true) as any,
    authorizeProjectToolRequest: (async () => ({ workspace: null })) as any,
    authorizeToolRequest: (_req, res, operation, options) => {
      expect(operation).toBe('project:visual-check');
      expect(options.endpoint).toBe('/api/projects/:id/visual-check');
      if (setup.grant === null) {
        res.status(401).json({ error: { code: 'TOOL_TOKEN_INVALID', message: 'bad token' } });
        return null;
      }
      return setup.grant ?? { projectId: 'p1', runId: 'run-1' };
    },
    projectDir: () => projectDir,
    assertProjectFile: async (_id, fileName) => {
      if (fileName === 'missing.html') throw Object.assign(new Error('nope'), { code: 'ENOENT' });
    },
    readRounds: async () => setup.rounds ?? 0,
    serviceDeps: () => ({
      render,
      createClient: async () => (setup.key === null ? null : ({ structured, resolveModel } as any)),
      readPrefs: async () => ({
        rounds: setup.rounds ?? 0,
        ...(setup.prefsModel === null ? {} : { model: setup.prefsModel ?? 'settings/model' }),
      }),
      readProjectPrompt: async () => setup.projectPrompt ?? null,
      now: () => new Date(Date.UTC(2026, 9, 28, 1, 2, 3, 4)),
    }),
  };
  const app = express();
  app.use(express.json());
  registerVisualCheckRoutes(app, deps);
  await new Promise<void>((resolve) => {
    server = app.listen(0, '127.0.0.1', () => resolve());
  });
  baseUrl = `http://127.0.0.1:${(server!.address() as { port: number }).port}`;
  return { structured, render };
}

function post(body: unknown, headers: Record<string, string> = {}) {
  return fetch(`${baseUrl}/api/projects/p1/visual-check`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
}

beforeEach(async () => {
  projectDir = await mkdtemp(path.join(tmpdir(), 'od-visual-check-'));
});

afterEach(async () => {
  if (server) await new Promise<void>((resolve) => server!.close(() => resolve()));
  server = undefined;
  await rm(projectDir, { recursive: true, force: true });
});

describe('POST /api/projects/:id/visual-check', () => {
  it('renders, grades against baseline + brief, saves PNGs under .visual-check, and reports usage', async () => {
    const { structured, render } = await start();
    const resp = await post({ fileName: 'pages/Pricing Page.html', brief: 'pricing page with three tiers', viewports: ['desktop', 'mobile'] });
    expect(resp.status).toBe(200);
    const body = (await resp.json()) as VisualCheckResponse;

    expect(render).toHaveBeenCalledWith({ fileName: 'pages/Pricing Page.html', viewports: ['desktop', 'mobile'] });
    const call = structured.mock.calls[0]![0];
    expect(call.model).toBe('settings/model');
    expect(call.content.filter((p: any) => p.type === 'image')).toHaveLength(2);
    const text = call.content.find((p: any) => p.type === 'text').text as string;
    expect(text).toContain(`b1: ${VISUAL_CHECK_BASELINE_CHECKLIST[0]}`);
    expect(text).toContain('pricing page with three tiers');

    expect(body.verdict).toBe('fail');
    expect(body.checklistSource).toBe('brief');
    expect(body.items.map((i) => i.id)).toEqual(['b1', 'b2', 'b3', 'b4', 'b5', 'd1']);
    expect(body.items[0]).toMatchObject({ pass: false, repairHint: 'Allow wrapping.', viewport: 'desktop', item: VISUAL_CHECK_BASELINE_CHECKLIST[0] });
    expect(body.items[1]!.repairHint).toBe('');
    expect(body.model).toBe('settings/model');
    expect(body.usage).toEqual({ inputTokens: 1200, outputTokens: 300 });
    expect(body.estimatedCostUsd).toBe(0.0012);
    expect(body.renderer).toBe('system-browser');
    expect(body.screenshots).toEqual([
      { viewport: 'desktop', width: 1440, height: 1200, bytes: 33, path: '.visual-check/Pricing_Page/20261028-010203-004-desktop.png' },
      { viewport: 'mobile', width: 390, height: 1800, bytes: 33, path: '.visual-check/Pricing_Page/20261028-010203-004-mobile.png' },
    ]);
    const saved = await readFile(path.join(projectDir, '.visual-check', 'Pricing_Page', '20261028-010203-004-desktop.png'));
    expect(saved.length).toBe(33);
    const record = JSON.parse(await readFile(path.join(projectDir, '.visual-check', 'Pricing_Page', '20261028-010203-004.json'), 'utf8'));
    expect(record.verdict).toBe('fail');

    const shot = await fetch(`${baseUrl}/api/projects/p1/visual-check/screenshot?path=${encodeURIComponent(body.screenshots[0]!.path)}`);
    expect(shot.status).toBe(200);
    expect(shot.headers.get('content-type')).toBe('image/png');
  });

  it('grades an explicit checklist on its own and passes when every item passes', async () => {
    const { structured } = await start({
      modelResponse: { summary: 'Fine.', items: [{ id: 'c1', item: 'x', pass: true, evidence: 'Fits.', repairHint: '' }] },
    });
    const body = (await (await post({ fileName: 'index.html', checklist: ['Headline fits on one line'], model: 'other/model' })).json()) as VisualCheckResponse;
    expect(body.verdict).toBe('pass');
    expect(body.checklistSource).toBe('request');
    expect(body.items).toEqual([{ id: 'c1', item: 'Headline fits on one line', pass: true, evidence: 'Fits.', repairHint: '' }]);
    expect(structured.mock.calls[0]![0].model).toBe('other/model');
    expect(structured.mock.calls[0]![0].content.at(-1).text).not.toContain('b1:');
  });

  it('writes the summary and every item in the requested language, translating the baseline', async () => {
    const { structured } = await start({
      modelResponse: {
        summary: 'ページは問題ありません。',
        items: [{ id: 'c1', item: '見出しが一行に収まる', pass: true, evidence: '収まっている。', repairHint: '' }],
      },
    });
    const body = (await (await post({ fileName: 'index.html', checklist: ['Headline fits on one line'], language: 'ja' })).json()) as VisualCheckResponse;
    const text = structured.mock.calls[0]![0].content.at(-1).text as string;
    expect(text).toMatch(/Write the summary, every item text, evidence and repairHint in Japanese/);
    expect(body.summary).toBe('ページは問題ありません。');
    expect(body.items[0]!.item).toBe('見出しが一行に収まる');
  });

  it('keeps the given requirement text when no language is asked for', async () => {
    const { structured } = await start({
      modelResponse: { summary: 'Fine.', items: [{ id: 'c1', item: 'something else', pass: true, evidence: 'Fits.', repairHint: '' }] },
    });
    const body = (await (await post({ fileName: 'index.html', checklist: ['Headline fits on one line'] })).json()) as VisualCheckResponse;
    expect(structured.mock.calls[0]![0].content.at(-1).text).not.toMatch(/Write the summary/);
    expect(body.items[0]!.item).toBe('Headline fits on one line');
  });

  it('falls back to the project prompt, then to the baseline only', async () => {
    await start({ projectPrompt: 'A launch page for a tea shop' });
    const withPrompt = (await (await post({ fileName: 'index.html' })).json()) as VisualCheckResponse;
    expect(withPrompt.checklistSource).toBe('project-prompt');
    server!.close();
    await start({ projectPrompt: null });
    const bare = (await (await post({ fileName: 'index.html' })).json()) as VisualCheckResponse;
    expect(bare.checklistSource).toBe('default');
    expect(bare.items.map((i) => i.id)).toEqual(['b1', 'b2', 'b3', 'b4', 'b5']);
  });

  it('turns a requirement the model skipped into a failure', async () => {
    await start({ modelResponse: { summary: '', items: [] } });
    const body = (await (await post({ fileName: 'index.html', checklist: ['A', 'B'] })).json()) as VisualCheckResponse;
    expect(body.verdict).toBe('fail');
    expect(body.items.every((i) => !i.pass && i.evidence.includes('no grade'))).toBe(true);
  });

  it('rejects bad requests before rendering', async () => {
    const { render } = await start();
    expect((await post({ fileName: 'index.html', apiKey: 'x' })).status).toBe(400);
    expect((await post({ fileName: 'logo.png' })).status).toBe(400);
    expect((await post({ fileName: 'missing.html' })).status).toBe(404);
    const other = await fetch(`${baseUrl}/api/projects/nope/visual-check`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{"fileName":"a.html"}' });
    expect(other.status).toBe(404);
    expect(render).not.toHaveBeenCalled();
  });

  it('reports a missing ai& key and an unavailable renderer as typed errors', async () => {
    await start({ key: null });
    const noKey = await post({ fileName: 'index.html' });
    expect(noKey.status).toBe(400);
    expect(((await noKey.json()) as any).error.code).toBe('VISUAL_CHECK_NOT_CONFIGURED');
    server!.close();
    await start({ renderError: new VisualCheckRenderError('no browser') });
    const noRender = await post({ fileName: 'index.html' });
    expect(noRender.status).toBe(503);
    expect(((await noRender.json()) as any).error.code).toBe('VISUAL_CHECK_RENDER_UNAVAILABLE');
  });

  it('caps a run-scoped tool token at the Settings rounds budget', async () => {
    const { structured } = await start({ rounds: 2 });
    const auth = { authorization: 'Bearer tool-token' };
    expect((await post({ fileName: 'index.html' }, auth)).status).toBe(200);
    expect((await post({ fileName: 'index.html' }, auth)).status).toBe(200);
    const third = await post({ fileName: 'index.html' }, auth);
    expect(third.status).toBe(429);
    expect(((await third.json()) as any).error.code).toBe('VISUAL_CHECK_BUDGET_EXHAUSTED');
    expect(structured).toHaveBeenCalledTimes(2);
    // The user's own checks (no tool token) are never budgeted.
    expect((await post({ fileName: 'index.html' })).status).toBe(200);
  });

  it('allows a run one check when rounds is 0, and refuses a token for another project', async () => {
    await start({ rounds: 0, grant: { projectId: 'p1', runId: 'run-0' } });
    const auth = { authorization: 'Bearer tool-token' };
    expect((await post({ fileName: 'index.html' }, auth)).status).toBe(200);
    expect((await post({ fileName: 'index.html' }, auth)).status).toBe(429);
    server!.close();
    await start({ grant: { projectId: 'p2', runId: 'run-x' } });
    expect((await post({ fileName: 'index.html' }, auth)).status).toBe(403);
  });

  it('picks a vision model from the ai& list when neither the request nor Settings names one', async () => {
    const { structured } = await start({ prefsModel: null });
    const resp = await post({ fileName: 'index.html' });
    expect(resp.status).toBe(200);
    expect(((await resp.json()) as VisualCheckResponse).model).toBe('catalog/vision');
    expect(structured.mock.calls[0]![0].model).toBe('catalog/vision');
  });

  it('rejects a model without vision or one the list lacks, before rendering and without spending budget', async () => {
    const { render, structured } = await start({ rounds: 1, prefsModel: 'text/only' });
    const auth = { authorization: 'Bearer tool-token' };
    const noVision = await post({ fileName: 'index.html' }, auth);
    expect(noVision.status).toBe(400);
    const error = ((await noVision.json()) as any).error;
    expect(error.code).toBe('VISUAL_CHECK_MODEL_INVALID');
    expect(error.message).toContain('"text/only" lacks vision');
    expect(error.message).toContain('catalog/vision, settings/model, other/model');
    const unknown = await post({ fileName: 'index.html', model: 'nope/missing' }, auth);
    expect(unknown.status).toBe(400);
    expect(((await unknown.json()) as any).error.message).toContain('"nope/missing" is not in the ai& model list');
    expect(render).not.toHaveBeenCalled();
    expect(structured).not.toHaveBeenCalled();
    // Both refusals were refunded, so the one budgeted check is still there.
    expect((await post({ fileName: 'index.html', model: 'other/model' }, auth)).status).toBe(200);
  });

  it('does not spend budget on a render failure', async () => {
    await start({ rounds: 1, renderError: new VisualCheckRenderError('no browser') });
    const auth = { authorization: 'Bearer tool-token' };
    expect((await post({ fileName: 'index.html' }, auth)).status).toBe(503);
    expect((await post({ fileName: 'index.html' }, auth)).status).toBe(503);
  });
});

describe('GET /api/projects/:id/visual-check/screenshot', () => {
  it.each(['../secret.png', '.visual-check/../../x.png', 'index.html', '.visual-check/a/b.json', '/etc/passwd'])(
    'rejects %s',
    async (bad) => {
      await start();
      const resp = await fetch(`${baseUrl}/api/projects/p1/visual-check/screenshot?path=${encodeURIComponent(bad)}`);
      expect(resp.status).toBe(400);
    },
  );
});

describe('project prompt lookup', () => {
  it('prefers the pending prompt, then the first user message of the oldest conversation', () => {
    expect(readProjectOriginalPrompt({ pendingPrompt: ' brief ' }, [], () => [])).toBe('brief');
    const messages: Record<string, Array<{ role: string; content: string }>> = {
      new: [{ role: 'user', content: 'later' }],
      old: [{ role: 'assistant', content: 'hi' }, { role: 'user', content: 'the original brief' }],
    };
    expect(readProjectOriginalPrompt(
      {},
      [{ id: 'new', createdAt: 2 }, { id: 'old', createdAt: 1 }],
      (id) => messages[id]!,
    )).toBe('the original brief');
    expect(readProjectOriginalPrompt(null, [], () => [])).toBeNull();
  });
});
