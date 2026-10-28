import { spawn } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import http from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const daemonRoot = fileURLToPath(new URL('..', import.meta.url));
const cliEntry = fileURLToPath(new URL('../src/cli.ts', import.meta.url));

let server: http.Server | undefined;
let baseUrl = '';
let seenBodies: string[] = [];
let tmp = '';

beforeEach(async () => {
  seenBodies = [];
  tmp = await mkdtemp(path.join(tmpdir(), 'od-comfy-cli-'));
  server = http.createServer((req, res) => {
    let body = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      if (req.method === 'POST' && (req.url ?? '').includes('/media/generate')) {
        seenBodies.push(body);
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ taskId: 'task-comfy', status: 'queued' }));
        return;
      }
      if ((req.url ?? '').includes('/media/tasks/')) {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ status: 'done', file: { name: 'out.png', size: 3 } }));
        return;
      }
      res.writeHead(404).end();
    });
  });
  await new Promise<void>((resolve) => server!.listen(0, '127.0.0.1', resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
});

afterEach(async () => {
  if (server) await new Promise<void>((resolve) => server!.close(() => resolve()));
  server = undefined;
  await rm(tmp, { recursive: true, force: true });
});

function runCli(extra: string[], model = 'comfyui-agent'): Promise<{ code: number; stderr: string }> {
  const args = ['--import', 'tsx', cliEntry, 'media', 'generate', '--surface', 'image', '--model', model, '--prompt', 'a lantern', '--daemon-url', baseUrl, ...extra];
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, {
      cwd: daemonRoot,
      env: { ...process.env, OD_PROJECT_ID: 'project-1', OD_TOOL_TOKEN: '' },
      stdio: ['ignore', 'ignore', 'pipe'],
    });
    let stderr = '';
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => (stderr += chunk));
    child.on('error', reject);
    child.on('close', (code) => resolve({ code: code ?? -1, stderr }));
  });
}

describe('od media generate ComfyUI flags', () => {
  it('sends size, transparency, negative, max rounds and the workflow file as body.comfy', async () => {
    const wf = path.join(tmp, 'wf.json');
    await writeFile(wf, JSON.stringify({ '1': { class_type: 'SaveImage', inputs: {} } }));
    const result = await runCli(['--size', '512x384', '--transparent', '--negative', 'text', '--max-rounds', '2', '--workflow', wf]);
    expect(result.code, result.stderr).toBe(0);
    const body = JSON.parse(seenBodies[0]!);
    expect(body.model).toBe('comfyui-agent');
    expect(body.comfy).toEqual({
      width: 512,
      height: 384,
      transparent: true,
      negative: 'text',
      maxRounds: 2,
      workflow: { '1': { class_type: 'SaveImage', inputs: {} } },
    });
  });

  it('omits body.comfy when no ComfyUI flag is used', async () => {
    const result = await runCli([], 'comfyui-workflow');
    expect(result.code, result.stderr).toBe(0);
    expect(JSON.parse(seenBodies[0]!).comfy).toBeUndefined();
  });

  it('rejects ComfyUI flags for other models and malformed values before any request', async () => {
    const other = await runCli(['--transparent'], 'gpt-image-2');
    expect(other.code).toBe(2);
    expect(other.stderr).toContain('only apply to comfyui-agent / comfyui-workflow');
    const size = await runCli(['--size', 'big']);
    expect(size.code).toBe(2);
    expect(size.stderr).toContain('--size must be WIDTHxHEIGHT');
    const rounds = await runCli(['--max-rounds', '0']);
    expect(rounds.code).toBe(2);
    const badJson = path.join(tmp, 'bad.json');
    await writeFile(badJson, '{nope');
    const wf = await runCli(['--workflow', badJson]);
    expect(wf.code).toBe(2);
    expect(wf.stderr).toContain('not valid JSON');
    expect(seenBodies).toHaveLength(0);
  });
});
