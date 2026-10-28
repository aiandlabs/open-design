import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import type http from 'node:http';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { startServer } from '../../src/server.js';

// End to end through the real daemon routes and the real `od` CLI: the policy
// is on by default (OD_MEDIA_LOCAL_ONLY unset), the catalogue shows readiness,
// and a hosted model is refused before any media task is created.

const daemonRoot = fileURLToPath(new URL('../..', import.meta.url));
const cliEntry = fileURLToPath(new URL('../../src/cli.ts', import.meta.url));

let server: http.Server | null = null;
let shutdown: (() => Promise<void> | void) | undefined;
let url = '';
let projectId = '';
let savedOverride: string | undefined;

beforeAll(async () => {
  savedOverride = process.env.OD_MEDIA_LOCAL_ONLY;
  delete process.env.OD_MEDIA_LOCAL_ONLY;
  const started = await startServer({ port: 0, returnServer: true }) as {
    url: string;
    server: http.Server;
    shutdown?: () => Promise<void> | void;
  };
  server = started.server;
  shutdown = started.shutdown;
  url = started.url;
  projectId = `project_${randomUUID()}`;
  const created = await fetch(`${url}/api/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: projectId, name: 'Local only', metadata: { kind: 'prototype' } }),
  });
  expect(created.status).toBe(200);
});

afterAll(async () => {
  await fetch(`${url}/api/app-config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ media: null }),
  }).catch(() => undefined);
  await Promise.resolve(shutdown?.());
  if (server) await new Promise<void>((resolve) => server?.close(() => resolve()));
  if (savedOverride === undefined) delete process.env.OD_MEDIA_LOCAL_ONLY;
  else process.env.OD_MEDIA_LOCAL_ONLY = savedOverride;
});

function runCli(args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const env: NodeJS.ProcessEnv = { ...process.env, OD_TOOL_TOKEN: '' };
    delete env.OD_MEDIA_LOCAL_ONLY;
    const child = spawn(process.execPath, ['--import', 'tsx', cliEntry, ...args, '--daemon-url', url], {
      cwd: daemonRoot,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => (stdout += chunk));
    child.stderr.on('data', (chunk) => (stderr += chunk));
    child.on('error', reject);
    child.on('close', (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

describe('local-only policy through the daemon and the CLI', () => {
  it('GET /api/media/models marks readiness and never defaults a disallowed model', async () => {
    const full = await (await fetch(`${url}/api/media/models`)).json() as any;
    expect(full.policy).toMatchObject({ localOnly: true, view: 'full', source: 'default' });
    expect(full.image.find((m: any) => m.id === 'vela/gpt-image-2')).toMatchObject({ allowed: false, ready: false });
    expect(full.image.find((m: any) => m.id === 'vela/gpt-image-2').default).toBeUndefined();
    expect(full.readiness.comfyui).toMatchObject({ local: true, allowed: true });

    const agent = await (await fetch(`${url}/api/media/models?view=agent`)).json() as any;
    expect(agent.image.map((m: any) => m.provider)).toEqual(['comfyui', 'comfyui']);
    expect(agent.video.map((m: any) => m.id)).toEqual(['hyperframes-html']);
  });

  it('POST media/generate refuses a hosted model with a typed 403', async () => {
    const response = await fetch(`${url}/api/projects/${projectId}/media/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ surface: 'image', model: 'vela/gpt-image-2', prompt: 'a lantern' }),
    });
    expect(response.status).toBe(403);
    const body = await response.json() as { error: { code: string; message: string } };
    expect(body.error.code).toBe('MEDIA_LOCAL_ONLY_SETUP_COMFYUI');
    expect(body.error.message).toContain('ComfyUI');
  });

  it('od media models --json shows the agent view; od media generate with a hosted model fails with the policy error', async () => {
    const models = await runCli(['media', 'models', '--json']);
    expect(models.code, models.stderr).toBe(0);
    const catalog = JSON.parse(models.stdout);
    expect(catalog.policy).toMatchObject({ localOnly: true, view: 'agent' });
    expect(catalog.image.every((m: any) => m.allowed)).toBe(true);

    const generated = await runCli([
      'media', 'generate', '--project', projectId, '--surface', 'image',
      '--model', 'vela/gpt-image-2', '--prompt', 'a lantern',
    ]);
    expect(generated.code).toBe(4);
    const line = generated.stderr.trim().split('\n').at(-1)!;
    expect(JSON.parse(line)).toMatchObject({
      error: { code: 'MEDIA_LOCAL_ONLY_SETUP_COMFYUI', nextStep: 'open-settings', retryable: false },
    });
  });

  it('od media policy reads and sets the same app-config key the Settings control writes', async () => {
    const shown = await runCli(['media', 'policy', '--json']);
    expect(shown.code, shown.stderr).toBe(0);
    expect(JSON.parse(shown.stdout)).toMatchObject({ localOnly: true, source: 'default', imageDefault: null });

    const off = await runCli(['media', 'policy', '--local-only', 'off', '--json']);
    expect(off.code, off.stderr).toBe(0);
    expect(JSON.parse(off.stdout)).toMatchObject({ localOnly: false, source: 'config' });
    const config = await (await fetch(`${url}/api/app-config`)).json() as any;
    expect(config.config.media).toEqual({ localOnly: false });

    const on = await runCli(['media', 'policy', '--local-only', 'on', '--json']);
    expect(JSON.parse(on.stdout)).toMatchObject({ localOnly: true, source: 'config' });
  });
});
