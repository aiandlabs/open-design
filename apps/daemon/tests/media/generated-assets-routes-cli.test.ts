import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import type http from 'node:http';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { startServer } from '../../src/server.js';

// Generated-asset review end to end: files and traces go in through the
// project file API, the review goes through the HTTP route and through the
// real `od` CLI, and deleting the image counts as a rejection.

const daemonRoot = fileURLToPath(new URL('../..', import.meta.url));
const cliEntry = fileURLToPath(new URL('../../src/cli.ts', import.meta.url));

let server: http.Server | null = null;
let shutdown: (() => Promise<void> | void) | undefined;
let url = '';
let projectId = '';

async function putFile(name: string, content: Buffer | string) {
  const body = Buffer.isBuffer(content)
    ? { name, content: content.toString('base64'), encoding: 'base64' }
    : { name, content };
  const res = await fetch(`${url}/api/projects/${projectId}/files`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  expect(res.status, await res.clone().text()).toBeLessThan(300);
}

beforeAll(async () => {
  const started = await startServer({ port: 0, returnServer: true }) as { url: string; server: http.Server; shutdown?: () => Promise<void> | void };
  server = started.server;
  shutdown = started.shutdown;
  url = started.url;
  projectId = `project_${randomUUID()}`;
  const created = await fetch(`${url}/api/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: projectId, name: 'Icons', metadata: { kind: 'prototype' } }),
  });
  expect(created.status).toBe(200);
  for (const name of ['leaf', 'rocket']) {
    await putFile(`${name}.png`, Buffer.from([0x89, 0x50, 0x4e, 0x47]));
    await putFile(`${name}.trace.json`, JSON.stringify({
      asset: `${name}.png`,
      set: 'onboarding',
      finishedAt: '2026-10-29T00:00:00.000Z',
      request: { prompt: `a ${name} icon`, width: 512, height: 512, kind: 'icon' },
    }));
  }
});

afterAll(async () => {
  await Promise.resolve(shutdown?.());
  if (server) await new Promise<void>((resolve) => server?.close(() => resolve()));
});

function runCli(args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['--import', 'tsx', cliEntry, ...args, '--daemon-url', url], {
      cwd: daemonRoot,
      env: { ...process.env, OD_TOOL_TOKEN: '' },
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

describe('generated asset review through the daemon and the CLI', () => {
  it('lists generated assets as accepted until reviewed', async () => {
    const res = await fetch(`${url}/api/projects/${projectId}/media/assets`);
    expect(res.status).toBe(200);
    const body = await res.json() as { assets: { path: string; kind?: string; set?: string; review: { state: string; reason: string } }[] };
    expect(body.assets.map((a) => [a.path, a.kind, a.set, a.review.state, a.review.reason])).toEqual([
      ['leaf.png', 'icon', 'onboarding', 'accepted', 'default'],
      ['rocket.png', 'icon', 'onboarding', 'accepted', 'default'],
    ]);
  });

  it('rejects and accepts through the HTTP route, and refuses bad input', async () => {
    const put = (body: unknown) => fetch(`${url}/api/projects/${projectId}/media/assets/review`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const rejected = await put({ path: 'rocket.png', state: 'rejected', via: 'ui' });
    expect(rejected.status).toBe(200);
    expect((await rejected.json() as any).asset.review).toMatchObject({ state: 'rejected', reason: 'user', via: 'ui' });
    expect((await put({ path: 'rocket.png', state: 'maybe' })).status).toBe(400);
    expect((await put({ path: 'nothing.png', state: 'rejected' })).status).toBe(404);
    expect((await put({ path: '../x.png', state: 'rejected' })).status).toBe(404);
  });

  it('od media review and od media assets use the same endpoints', async () => {
    const accepted = await runCli(['media', 'review', 'rocket.png', '--accept', '--project', projectId, '--json']);
    expect(accepted.code, accepted.stderr).toBe(0);
    expect(JSON.parse(accepted.stdout).asset.review).toMatchObject({ state: 'accepted', via: 'cli' });

    const rejected = await runCli(['media', 'review', 'leaf.png', '--reject', '--project', projectId]);
    expect(rejected.code, rejected.stderr).toBe(0);
    expect(rejected.stdout).toContain('rejected leaf.png');

    const listed = await runCli(['media', 'assets', '--project', projectId, '--json']);
    expect(listed.code, listed.stderr).toBe(0);
    expect(JSON.parse(listed.stdout).assets.map((a: any) => [a.path, a.review.state])).toEqual([
      ['leaf.png', 'rejected'],
      ['rocket.png', 'accepted'],
    ]);

    const usage = await runCli(['media', 'review', 'leaf.png', '--project', projectId]);
    expect(usage.code).toBe(2);
    expect(usage.stderr).toContain('--accept | --reject');
  });

  it('counts a deleted image as rejected', async () => {
    const del = await fetch(`${url}/api/projects/${projectId}/files/rocket.png`, { method: 'DELETE' });
    expect(del.status).toBeLessThan(300);
    const body = await (await fetch(`${url}/api/projects/${projectId}/media/assets`)).json() as any;
    expect(body.assets.find((a: any) => a.path === 'rocket.png')).toMatchObject({ exists: false, review: { state: 'rejected', reason: 'deleted' } });
  });
});
