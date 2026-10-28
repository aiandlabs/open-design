import { spawn } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import http from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const daemonRoot = fileURLToPath(new URL('..', import.meta.url));
const cliEntry = fileURLToPath(new URL('../src/cli.ts', import.meta.url));

const RESULT = {
  verdict: 'fail',
  summary: 'The heading is clipped.',
  items: [
    { id: 'b1', item: 'All text is fully visible', pass: false, evidence: 'Heading runs off the right edge.', repairHint: 'Let it wrap.', viewport: 'desktop' },
    { id: 'b2', item: 'Contrast is readable', pass: true, evidence: 'Dark on white.', repairHint: '' },
  ],
  checklistSource: 'brief',
  screenshots: [{ viewport: 'desktop', width: 1440, height: 900, path: '.visual-check/index/x-desktop.png', bytes: 10 }],
  renderer: 'system-browser',
  model: 'zai-org/glm-5.3-flash',
  apiShape: 'openai',
  usage: { inputTokens: 1000, outputTokens: 200 },
  durationMs: 1500,
};

let server: http.Server | undefined;
let baseUrl = '';
let seen: Array<{ url: string; body: string; auth: string | undefined }> = [];
let tmp = '';
let failWith: { status: number; body: unknown } | null = null;

beforeEach(async () => {
  seen = [];
  failWith = null;
  tmp = await mkdtemp(path.join(tmpdir(), 'od-visual-check-cli-'));
  server = http.createServer((req, res) => {
    let body = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      if (req.method === 'POST' && (req.url ?? '').endsWith('/visual-check')) {
        seen.push({ url: req.url ?? '', body, auth: req.headers.authorization });
        if (failWith) {
          res.writeHead(failWith.status, { 'content-type': 'application/json' });
          res.end(JSON.stringify(failWith.body));
          return;
        }
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(RESULT));
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

function runCli(extra: string[], opts: { stdin?: string; env?: Record<string, string> } = {}): Promise<{ code: number; stdout: string; stderr: string }> {
  const args = ['--import', 'tsx', cliEntry, 'visual-check', ...extra, '--daemon-url', baseUrl];
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, {
      cwd: daemonRoot,
      env: { ...process.env, OD_PROJECT_ID: 'project-1', OD_TOOL_TOKEN: '', ...opts.env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => (stdout += chunk));
    child.stderr.on('data', (chunk) => (stderr += chunk));
    child.on('error', reject);
    child.on('close', (code) => resolve({ code: code ?? -1, stdout, stderr }));
    child.stdin.end(opts.stdin ?? '');
  });
}

describe('od visual-check', () => {
  it('posts file, brief and viewports to the project endpoint and prints the grades', async () => {
    const result = await runCli(['index.html', '--brief', 'pricing page', '--viewport', 'both']);
    expect(result.code, result.stderr).toBe(0);
    expect(seen).toHaveLength(1);
    expect(seen[0]!.url).toBe('/api/projects/project-1/visual-check');
    expect(JSON.parse(seen[0]!.body)).toEqual({ fileName: 'index.html', viewports: ['desktop', 'mobile'], brief: 'pricing page' });
    expect(result.stdout).toContain('verdict: FAIL  (1/2 passed)');
    expect(result.stdout).toContain('[FAIL] b1: All text is fully visible');
    expect(result.stdout).toContain('fix: Let it wrap.');
    expect(result.stdout).toContain('[PASS] b2');
    expect(result.stdout).toContain('model: zai-org/glm-5.3-flash (openai) · tokens: 1000 in / 200 out');
  });

  it('prints the raw result with --json and sends the run tool token', async () => {
    const result = await runCli(['index.html', '--project', 'p9', '--json'], { env: { OD_TOOL_TOKEN: 'tok-1' } });
    expect(result.code, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual(RESULT);
    expect(seen[0]!.url).toBe('/api/projects/p9/visual-check');
    expect(seen[0]!.auth).toBe('Bearer tok-1');
    expect(JSON.parse(seen[0]!.body)).toEqual({ fileName: 'index.html', viewports: ['desktop'] });
  });

  it('reads the brief from --prompt-file - and the checklist from a file', async () => {
    const list = path.join(tmp, 'checks.txt');
    await writeFile(list, '- Hero fits on one line\n\n2. CTA is visible\n');
    const result = await runCli(['index.html', '--prompt-file', '-', '--checklist-file', list, '--model', 'm/x', '--viewport', 'mobile'], { stdin: 'a tea shop launch page\n' });
    expect(result.code, result.stderr).toBe(0);
    expect(JSON.parse(seen[0]!.body)).toEqual({
      fileName: 'index.html',
      viewports: ['mobile'],
      brief: 'a tea shop launch page',
      checklist: ['Hero fits on one line', 'CTA is visible'],
      model: 'm/x',
    });
  });

  it('rejects bad usage before any request', async () => {
    expect((await runCli(['index.html', '--viewport', 'tablet'])).code).toBe(2);
    expect((await runCli(['index.html', '--brief', 'a', '--prompt-file', 'b'])).code).toBe(2);
    const noProject = await runCli(['index.html'], { env: { OD_PROJECT_ID: '' } });
    expect(noProject.code).toBe(2);
    expect(noProject.stderr).toContain('--project');
    expect(seen).toHaveLength(0);
  });

  it('surfaces daemon errors as a non-zero exit', async () => {
    failWith = { status: 429, body: { error: { code: 'VISUAL_CHECK_BUDGET_EXHAUSTED', message: 'budget used' } } };
    const result = await runCli(['index.html']);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain('budget used');
  });
});
