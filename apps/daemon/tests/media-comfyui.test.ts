import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { createImage, decodePng, encodePng } from '@open-design/comfy-agent';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { hasStoredProviderEntry } from '../src/media/config.js';
import { comfySizeForAspect, parseComfyMediaOptions } from '../src/media/comfyui.js';
import { generateMedia } from '../src/media/index.js';
import { findMediaModel, findProvider } from '../src/media/models.js';
import { mediaTaskErrorFromFailure } from '../src/media/task-error.js';
import { renderMediaGenerationContract } from '../src/prompts/media-contract.js';

// One local server stands in for both ComfyUI (no websocket, so the client
// polls /history) and the ai& API (`AIAND_BASE_URL` points at /v1 here).

type Graph = Record<string, { class_type: string; inputs: Record<string, unknown> }>;

const node = (required: Record<string, unknown[]>, output: string[], outputNode = false) => ({
  input: { required },
  input_order: { required: Object.keys(required) },
  output,
  name: 'x',
  output_node: outputNode,
});

const OBJECT_INFO = {
  CheckpointLoaderSimple: node({ ckpt_name: [['sd_xl_base_1.0.safetensors'], {}] }, ['MODEL', 'CLIP', 'VAE']),
  CLIPTextEncode: node({ text: ['STRING', {}], clip: ['CLIP', {}] }, ['CONDITIONING']),
  EmptyLatentImage: node({ width: ['INT', { min: 16, max: 16384 }], height: ['INT', { min: 16, max: 16384 }], batch_size: ['INT', { min: 1, max: 64 }] }, ['LATENT']),
  KSampler: node(
    {
      model: ['MODEL', {}],
      seed: ['INT', { min: 0, max: 18446744073709551615, control_after_generate: true }],
      steps: ['INT', { min: 1, max: 10000 }],
      cfg: ['FLOAT', { min: 0, max: 100 }],
      sampler_name: [['euler', 'dpmpp_2m'], {}],
      scheduler: [['normal', 'karras'], {}],
      positive: ['CONDITIONING', {}],
      negative: ['CONDITIONING', {}],
      latent_image: ['LATENT', {}],
      denoise: ['FLOAT', { min: 0, max: 1 }],
    },
    ['LATENT'],
  ),
  VAEDecode: node({ samples: ['LATENT', {}], vae: ['VAE', {}] }, ['IMAGE']),
  SaveImage: node({ images: ['IMAGE', {}], filename_prefix: ['STRING', {}] }, [], true),
};

let server: http.Server;
let baseUrl = '';
let prompts: Graph[] = [];
let llmCalls: { tool: string; model: string }[] = [];
let root = '';
let projectRoot = '';
let projectsRoot = '';
const savedEnv: Record<string, string | undefined> = {};
const ENV_KEYS = ['OD_MEDIA_CONFIG_DIR', 'OD_DATA_DIR', 'AIAND_API_KEY', 'OD_COMFYUI_AIAND_API_KEY', 'AIAND_BASE_URL', 'COMFYUI_URL', 'OD_MEDIA_ALLOW_STUBS'];

function render(graph: Graph): Buffer {
  const latent = Object.values(graph).find((n) => n.class_type === 'EmptyLatentImage');
  const w = Number(latent?.inputs.width ?? 64);
  const h = Number(latent?.inputs.height ?? 64);
  return encodePng(createImage(w, h, { r: 220, g: 40, b: 40 }));
}

function json(res: http.ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

beforeEach(async () => {
  prompts = [];
  llmCalls = [];
  const history = new Map<string, unknown>();
  const files = new Map<string, Buffer>();
  server = http.createServer((req, res) => {
    let body = '';
    req.setEncoding('utf8');
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      const url = new URL(req.url ?? '/', 'http://x');
      if (url.pathname === '/object_info') return json(res, 200, OBJECT_INFO);
      if (req.method === 'POST' && url.pathname === '/prompt') {
        const graph = JSON.parse(body).prompt as Graph;
        prompts.push(graph);
        const id = randomUUID();
        const filename = `out-${prompts.length}.png`;
        files.set(filename, render(graph));
        const saveId = Object.entries(graph).find(([, n]) => n.class_type === 'SaveImage')![0];
        history.set(id, { outputs: { [saveId]: { images: [{ filename, subfolder: '', type: 'output' }] } }, status: { status_str: 'success', completed: true, messages: [] } });
        return json(res, 200, { prompt_id: id, number: prompts.length, node_errors: {} });
      }
      if (url.pathname.startsWith('/history/')) {
        const id = url.pathname.slice('/history/'.length);
        return json(res, 200, history.has(id) ? { [id]: history.get(id) } : {});
      }
      if (url.pathname === '/view') {
        const bytes = files.get(url.searchParams.get('filename') ?? '');
        if (!bytes) return json(res, 404, {});
        res.writeHead(200, { 'content-type': 'image/png' });
        res.end(bytes);
        return;
      }
      if (url.pathname === '/v1/models') {
        return json(res, 200, { data: [
          { id: 'test/text', capabilities: ['chat', 'tool_calling'], input_per_1m: '1', output_per_1m: '2' },
          { id: 'test/planner', capabilities: ['vision', 'tool_calling'], input_per_1m: '1', output_per_1m: '2' },
          { id: 'test/critic', capabilities: ['vision', 'tool_calling'], input_per_1m: '1', output_per_1m: '2' },
        ] });
      }
      if (url.pathname === '/v1/chat/completions') {
        const req = JSON.parse(body) as { model: string; tool_choice: { function: { name: string } } };
        const tool = req.tool_choice.function.name;
        llmCalls.push({ tool, model: req.model });
        const args = tool === 'submit_plan'
          ? { checklist: [{ id: 'subject', text: 'One red square' }], positive_prompt: 'a red square', negative_prompt: 'text' }
          : { items: [{ id: 'subject', pass: true, evidence: 'red square' }], overall: 'ok' };
        return json(res, 200, {
          model: req.model,
          choices: [{ message: { tool_calls: [{ type: 'function', function: { name: tool, arguments: JSON.stringify(args) } }] } }],
          usage: { prompt_tokens: 100, completion_tokens: 10 },
        });
      }
      json(res, 404, {});
    });
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

  root = await mkdtemp(path.join(tmpdir(), 'od-comfyui-'));
  projectRoot = path.join(root, 'project-root');
  projectsRoot = path.join(projectRoot, '.od', 'projects');
  await mkdir(projectsRoot, { recursive: true });
  for (const k of ENV_KEYS) {
    savedEnv[k] = process.env[k];
    delete process.env[k];
  }
  process.env.AIAND_BASE_URL = `${baseUrl}/v1`;
});

afterEach(async () => {
  await new Promise<void>((r) => server.close(() => r()));
  for (const k of ENV_KEYS) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k];
  }
  await rm(root, { recursive: true, force: true });
});

async function writeMediaConfig(entry: Record<string, string>) {
  const file = path.join(projectRoot, '.od', 'media-config.json');
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify({ providers: { comfyui: entry } }), 'utf8');
}

describe('comfyui provider registry', () => {
  it('is an integrated provider with two image models', () => {
    expect(findProvider('comfyui')).toMatchObject({ integrated: true, defaultBaseUrl: 'http://127.0.0.1:8188' });
    expect(findMediaModel('comfyui-agent')).toMatchObject({ provider: 'comfyui' });
    expect(findMediaModel('comfyui-workflow')).toMatchObject({ provider: 'comfyui' });
    // Existing providers are untouched.
    expect(findMediaModel('minimax-image-01')).toMatchObject({ provider: 'minimax' });
    expect(findProvider('minimax')).toMatchObject({ integrated: true });
  });
});

describe('generateMedia with comfyui-workflow', () => {
  it('runs the default base graph, sizes the output exactly, and writes a trace beside it', async () => {
    await writeMediaConfig({ baseUrl });
    const result = await generateMedia({
      projectRoot,
      projectsRoot,
      projectId: 'p1',
      surface: 'image',
      model: 'comfyui-workflow',
      prompt: 'a red square',
      output: 'square.png',
      comfy: { width: 512, height: 512, seed: 9 },
    });
    expect(result).toMatchObject({ name: 'square.png', providerId: 'comfyui', usedStubFallback: false });
    expect(result.providerNote).toContain('comfyui/comfyui-workflow');
    const dir = path.join(projectsRoot, 'p1');
    const png = decodePng(await readFile(path.join(dir, 'square.png')));
    expect([png.width, png.height]).toEqual([512, 512]);
    // SDXL at 512² generates at 768² (smallest size the family handles well).
    expect(prompts[0]!.latent!.inputs).toMatchObject({ width: 768, height: 768 });
    expect(prompts[0]!.sampler!.inputs.seed).toBe(9);
    const trace = JSON.parse(await readFile(path.join(dir, 'square.trace.json'), 'utf8'));
    expect(trace).toMatchObject({ kind: 'comfyui-workflow', promptId: expect.any(String) });
    expect(trace.graph.pos.inputs.text).toBe('a red square');
  });

  it('fills prompt, size and seed slots in a user-supplied API graph', async () => {
    await writeMediaConfig({ baseUrl });
    const workflow: Graph = {
      '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sd_xl_base_1.0.safetensors' } },
      '6': { class_type: 'CLIPTextEncode', inputs: { text: 'placeholder', clip: ['4', 1] } },
      '7': { class_type: 'CLIPTextEncode', inputs: { text: 'bad', clip: ['4', 1] } },
      '5': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
      '3': { class_type: 'KSampler', inputs: { model: ['4', 0], seed: 1, steps: 4, cfg: 5, sampler_name: 'euler', scheduler: 'normal', positive: ['6', 0], negative: ['7', 0], latent_image: ['5', 0], denoise: 1 } },
      '8': { class_type: 'VAEDecode', inputs: { samples: ['3', 0], vae: ['4', 2] } },
      '9': { class_type: 'SaveImage', inputs: { images: ['8', 0], filename_prefix: 'mine' } },
    };
    await generateMedia({
      projectRoot, projectsRoot, projectId: 'p1', surface: 'image', model: 'comfyui-workflow',
      prompt: 'filled prompt', output: 'w.png', aspect: '16:9', comfy: { workflow, seed: 42 },
    });
    expect(prompts[0]!['6']!.inputs.text).toBe('filled prompt');
    expect(prompts[0]!['3']!.inputs).toMatchObject({ seed: 42, steps: 4 });
    expect(prompts[0]!['5']!.inputs).toMatchObject({ width: 1344, height: 768 });
  });
});

describe('generateMedia with comfyui-agent', () => {
  it('runs the planner/critic loop with the configured models and writes the trace', async () => {
    await writeMediaConfig({ baseUrl, apiKey: 'test-aiand-key', model: 'planner=test/planner, critic=test/critic, shape=openai' });
    const result = await generateMedia({
      projectRoot, projectsRoot, projectId: 'p1', surface: 'image', model: 'comfyui-agent',
      prompt: 'a red square', output: 'agent.png', comfy: { width: 256, height: 256, maxRounds: 2 },
    });
    expect(result.name).toBe('agent.png');
    expect(result.providerNote).toMatch(/comfyui\/comfyui-agent .* 1 round\(s\) .* 2\/2 checks .* all-pass/);
    // Planning and grading, in order; the blind recognition check (submit_recognition,
    // then submit_match) runs on the critic model after grading.
    expect(llmCalls.filter((c) => c.tool === 'submit_plan' || c.tool === 'submit_grades')).toEqual([
      { tool: 'submit_plan', model: 'test/planner' },
      { tool: 'submit_grades', model: 'test/critic' },
    ]);
    expect(llmCalls.filter((c) => c.tool === 'submit_recognition').every((c) => c.model === 'test/critic')).toBe(true);
    const dir = path.join(projectsRoot, 'p1');
    const trace = JSON.parse(await readFile(path.join(dir, 'agent.trace.json'), 'utf8'));
    expect(trace).toMatchObject({ stopReason: 'all-pass', asset: 'agent.png', models: { planner: 'test/planner', critic: 'test/critic' } });
    // Trace image paths are project-relative and exist.
    expect(trace.rounds[0].imagePath).toBe('.comfy-agent/agent/round-1.png');
    expect(existsSync(path.join(dir, trace.rounds[0].imagePath))).toBe(true);
    expect(decodePng(await readFile(path.join(dir, 'agent.png')))).toMatchObject({ width: 256, height: 256 });
  });

  it('picks the models from the ai& list when the model field is blank', async () => {
    await writeMediaConfig({ baseUrl, apiKey: 'test-aiand-key' });
    await generateMedia({
      projectRoot, projectsRoot, projectId: 'p1', surface: 'image', model: 'comfyui-agent',
      prompt: 'a red square', output: 'auto.png', comfy: { width: 256, height: 256, maxRounds: 1 },
    });
    // No preferred id is listed here, so both roles take the first model with vision + tool calling.
    expect(llmCalls.filter((c) => c.tool === 'submit_plan' || c.tool === 'submit_grades').map((c) => c.model)).toEqual(['test/planner', 'test/planner']);
  });

  it('refuses a configured critic without vision as a 400, before ComfyUI runs', async () => {
    await writeMediaConfig({ baseUrl, apiKey: 'test-aiand-key', model: 'planner=test/planner, critic=test/text' });
    const err = await generateMedia({ projectRoot, projectsRoot, projectId: 'p1', surface: 'image', model: 'comfyui-agent', prompt: 'x' }).catch((e: unknown) => e);
    expect(err).toMatchObject({ code: 'LLM_MODEL', status: 400 });
    expect((err as Error).message).toContain('critic model "test/text" lacks vision');
    expect((err as Error).message).toContain('valid critic models: test/planner, test/critic');
    expect(llmCalls).toHaveLength(0);
    expect(prompts).toHaveLength(0);
  });

  it('asks for the ai& key in Settings when none is configured', async () => {
    await writeMediaConfig({ baseUrl });
    const err = await generateMedia({ projectRoot, projectsRoot, projectId: 'p1', surface: 'image', model: 'comfyui-agent', prompt: 'x' }).catch((e: unknown) => e);
    expect((err as Error & { code?: string }).code).toBe('LLM_CONFIG');
    expect((err as Error).message).toMatch(/ai& API key/);
    expect(mediaTaskErrorFromFailure(err, { model: 'comfyui-agent' }).nextStep).toBe('open-settings');
    expect(prompts).toHaveLength(0);
  });

  it('generates an icon for a set with the project\'s accepted icons, and records why references were not used on this model', async () => {
    await writeMediaConfig({ baseUrl, apiKey: 'test-aiand-key', model: 'planner=test/planner, critic=test/critic' });
    const dir = path.join(projectsRoot, 'p1');
    await mkdir(path.join(dir, 'icons'), { recursive: true });
    for (const [name, review] of [['leaf', undefined], ['rocket', { state: 'rejected' }]] as const) {
      await writeFile(path.join(dir, 'icons', `${name}.png`), encodePng(createImage(32, 32, { r: 40, g: 160, b: 60 })));
      await writeFile(path.join(dir, 'icons', `${name}.trace.json`), JSON.stringify({
        asset: `icons/${name}.png`, set: 'onboarding', finishedAt: '2026-10-29T00:00:00.000Z',
        request: { prompt: name, width: 32, height: 32, kind: 'icon' }, ...(review ? { review } : {}),
      }));
    }
    await generateMedia({
      projectRoot, projectsRoot, projectId: 'p1', surface: 'image', model: 'comfyui-agent',
      prompt: 'a coffee cup icon', output: 'coffee.png',
      comfy: { width: 256, height: 256, maxRounds: 1, kind: 'icon', set: 'onboarding' },
      referenceCacheDir: path.join(root, 'reference-cache'),
    });
    const trace = JSON.parse(await readFile(path.join(dir, 'coffee.trace.json'), 'utf8'));
    expect(trace).toMatchObject({ asset: 'coffee.png', set: 'onboarding', request: { kind: 'icon' } });
    // The fake server only has SDXL, which cannot take reference images.
    expect(trace.references).toMatchObject({ shape: [], set: [], note: expect.stringMatching(/cannot take reference images/) });
    expect(Object.values(prompts[0]!).some((n) => n.class_type === 'ReferenceLatent')).toBe(false);
  });

  it('reports an unreachable server as a connection error, never a stub, even when stubs are allowed', async () => {
    process.env.OD_MEDIA_ALLOW_STUBS = '1';
    await writeMediaConfig({ baseUrl: 'http://127.0.0.1:1' });
    const err = await generateMedia({ projectRoot, projectsRoot, projectId: 'p1', surface: 'image', model: 'comfyui-workflow', prompt: 'x', output: 'none.png' }).catch((e: unknown) => e);
    expect((err as Error & { code?: string; status?: number }).code).toBe('COMFY_CONNECTION');
    expect((err as Error & { status?: number }).status).toBe(503);
    expect(mediaTaskErrorFromFailure(err, { model: 'comfyui-workflow' }).nextStep).toBe('open-settings');
    expect(existsSync(path.join(projectsRoot, 'p1', 'none.png'))).toBe(false);
  });
});

describe('comfyui request options and prompt gating', () => {
  it('parses the comfy request options strictly', () => {
    expect(parseComfyMediaOptions({ comfy: { width: 512, height: 512, transparent: true, maxRounds: 2, negative: ' text ', workflow: { a: 1 } } }))
      .toEqual({ width: 512, height: 512, transparent: true, maxRounds: 2, negative: 'text', workflow: { a: 1 } });
    expect(parseComfyMediaOptions({ comfy: { kind: 'icon', set: ' Onboarding icons ', references: false } }))
      .toEqual({ kind: 'icon', set: 'Onboarding icons', references: false });
    expect(parseComfyMediaOptions({ comfy: { kind: 'poster', set: 'x'.repeat(65) } })).toBeUndefined();
    expect(parseComfyMediaOptions({ comfy: { width: 512 } })).toBeUndefined();
    expect(parseComfyMediaOptions({ comfy: { maxRounds: 99 } })).toBeUndefined();
    expect(parseComfyMediaOptions({})).toBeUndefined();
    expect(comfySizeForAspect('9:16')).toEqual({ width: 768, height: 1344 });
    expect(comfySizeForAspect(undefined)).toEqual({ width: 1024, height: 1024 });
  });

  it('adds the ComfyUI section to the media contract only when the provider is configured', async () => {
    expect(renderMediaGenerationContract(undefined, undefined)).not.toContain('Image generation (local ComfyUI)');
    const on = renderMediaGenerationContract(undefined, undefined, {
      localImageGeneration: { localOnly: false, comfyuiModel: 'comfyui-agent' },
    });
    expect(on).toContain('### Image generation (local ComfyUI)');
    expect(on.indexOf('Image generation (local ComfyUI)')).toBeLessThan(on.indexOf('### Allowed model IDs'));
    expect(on).toContain('--model comfyui-agent');

    expect(await hasStoredProviderEntry(projectRoot, 'comfyui')).toBe(false);
    process.env.AIAND_API_KEY = 'env-only';
    expect(await hasStoredProviderEntry(projectRoot, 'comfyui')).toBe(false);
    await writeMediaConfig({ baseUrl });
    expect(await hasStoredProviderEntry(projectRoot, 'comfyui')).toBe(true);
  });
});
