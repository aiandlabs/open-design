import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { mediaFailureNextStep } from '@open-design/contracts';

import { readAppConfig, writeAppConfig } from '../../src/app-config.js';
import { generateMedia } from '../../src/media/index.js';
import {
  MediaLocalOnlyError,
  buildMediaCatalog,
  readMediaLocalOnly,
  resolveComfyImageModel,
  resolveMediaLocalOnlyState,
} from '../../src/media/local-only.js';
import { mediaTaskErrorFromFailure } from '../../src/media/task-error.js';

// Process-wide env these tests depend on. The daemon test setup turns the
// policy off with OD_MEDIA_LOCAL_ONLY=0 for the upstream suites; every test
// here starts from "unset" and says what it wants.
const ENV_KEYS = [
  'OD_MEDIA_LOCAL_ONLY',
  'OD_MEDIA_CONFIG_DIR',
  'OD_DATA_DIR',
  'AIAND_API_KEY',
  'OD_COMFYUI_AIAND_API_KEY',
  'OD_MEDIA_ALLOW_STUBS',
  'OPENAI_API_KEY',
  'OD_OPENAI_API_KEY',
] as const;
const savedEnv: Record<string, string | undefined> = {};

let root = '';
let projectRoot = '';
let projectsRoot = '';
let dataDir = '';

beforeEach(async () => {
  for (const key of ENV_KEYS) {
    savedEnv[key] = process.env[key];
    delete process.env[key];
  }
  root = await mkdtemp(path.join(tmpdir(), 'od-local-only-'));
  projectRoot = path.join(root, 'project-root');
  projectsRoot = path.join(projectRoot, '.od', 'projects');
  dataDir = path.join(root, 'data');
  await mkdir(projectsRoot, { recursive: true });
  await mkdir(dataDir, { recursive: true });
});

afterEach(async () => {
  for (const key of ENV_KEYS) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
  await rm(root, { recursive: true, force: true });
});

async function writeMediaConfig(providers: Record<string, Record<string, string>>) {
  const file = path.join(projectRoot, '.od', 'media-config.json');
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify({ providers }), 'utf8');
}

function generate(model: string, surface: 'image' | 'video' | 'audio' = 'image', localOnly?: boolean) {
  return generateMedia({
    projectRoot,
    projectsRoot,
    projectId: 'p1',
    surface,
    model,
    prompt: 'a red paper lantern',
    output: 'out.png',
    ...(localOnly === undefined ? {} : { localOnly }),
  });
}

async function refusal(promise: Promise<unknown>): Promise<MediaLocalOnlyError> {
  const err = await promise.then(
    () => { throw new Error('expected a refusal'); },
    (e: unknown) => e,
  );
  expect(err).toBeInstanceOf(MediaLocalOnlyError);
  return err as MediaLocalOnlyError;
}

describe('policy resolution', () => {
  it('defaults to on, follows app-config, and the env override wins', async () => {
    expect(resolveMediaLocalOnlyState(null)).toEqual({ localOnly: true, source: 'default' });
    expect(resolveMediaLocalOnlyState({ media: { localOnly: false } })).toEqual({ localOnly: false, source: 'config' });
    process.env.OD_MEDIA_LOCAL_ONLY = '1';
    expect(resolveMediaLocalOnlyState({ media: { localOnly: false } })).toEqual({ localOnly: true, source: 'env' });
    process.env.OD_MEDIA_LOCAL_ONLY = '0';
    expect(resolveMediaLocalOnlyState(null)).toEqual({ localOnly: false, source: 'env' });
  });

  it('stores media.localOnly in app-config and refuses a malformed value', async () => {
    expect((await readMediaLocalOnly(dataDir)).localOnly).toBe(true);
    await writeAppConfig(dataDir, { media: { localOnly: false } });
    expect((await readAppConfig(dataDir)).media).toEqual({ localOnly: false });
    expect(await readMediaLocalOnly(dataDir)).toEqual({ localOnly: false, source: 'config' });
    await expect(writeAppConfig(dataDir, { media: { localOnly: 'no' } } as never)).rejects.toMatchObject({
      code: 'INVALID_APP_CONFIG_VALUE',
    });
    await writeAppConfig(dataDir, { media: null });
    expect(await readMediaLocalOnly(dataDir)).toEqual({ localOnly: true, source: 'default' });
  });

  it('reads an unreadable app-config as on, never off', async () => {
    await writeFile(path.join(dataDir, 'app-config.json'), '{not json', 'utf8');
    expect((await readMediaLocalOnly(dataDir)).localOnly).toBe(true);
  });
});

describe('generateMedia refuses hosted image and video providers', () => {
  it.each([
    ['vela/gpt-image-2'],
    ['nano-banana'],
    ['gpt-image-2'],
    ['gemini-3.1-flash-image-preview'],
    ['openrouter/google/gemini-2.5-flash-image'],
    ['fal-ai/flux/dev'],
    ['aihubmix-gpt-image-1'],
    ['custom-image'],
    ['flux-pro-ultra'],
  ])('%s, with ComfyUI not configured: set up ComfyUI', async (model) => {
    const err = await refusal(generate(model, 'image', true));
    expect(err.code).toBe('MEDIA_LOCAL_ONLY_SETUP_COMFYUI');
    expect(err.status).toBe(403);
    expect(err.message).toContain('Settings → Media providers → ComfyUI');
    expect(mediaTaskErrorFromFailure(err, { model }).nextStep).toBe('open-settings');
  });

  it('points at comfyui-agent when ComfyUI is configured with the ai& key', async () => {
    await writeMediaConfig({ comfyui: { baseUrl: 'http://127.0.0.1:1', apiKey: 'aiand-key' } });
    const err = await refusal(generate('vela/gpt-image-2', 'image', true));
    expect(err.code).toBe('MEDIA_LOCAL_ONLY_SWITCH_MODEL');
    expect(err.localModel).toBe('comfyui-agent');
    expect(err.message).toContain('--model comfyui-agent');
    expect(mediaFailureNextStep({ code: err.code })).toBe('switch-model');
  });

  it('points at comfyui-workflow when the ai& key is missing', async () => {
    await writeMediaConfig({ comfyui: { baseUrl: 'http://127.0.0.1:1' } });
    expect(await resolveComfyImageModel(projectRoot)).toBe('comfyui-workflow');
    const err = await refusal(generate('gpt-image-2', 'image', true));
    expect(err.localModel).toBe('comfyui-workflow');
  });

  it('refuses hosted video with no local substitute', async () => {
    const err = await refusal(generate('doubao-seedance-2-0-260128', 'video', true));
    expect(err.code).toBe('MEDIA_LOCAL_ONLY_VIDEO');
    expect(mediaFailureNextStep({ code: err.code })).toBe('unsupported');
  });

  it('is fail-closed when the caller passes no policy and no override is set', async () => {
    const err = await refusal(generate('vela/gpt-image-2'));
    expect(err.code).toBe('MEDIA_LOCAL_ONLY_SETUP_COMFYUI');
  });

  it('lets the local ComfyUI provider and audio through the policy', async () => {
    await writeMediaConfig({ comfyui: { baseUrl: 'http://127.0.0.1:1' } });
    // Port 1 is closed: the call gets past the policy and fails on the
    // connection instead.
    const comfy = await generate('comfyui-workflow', 'image', true).catch((e: unknown) => e);
    expect(comfy).not.toBeInstanceOf(MediaLocalOnlyError);
    expect((comfy as { code?: string }).code).toBe('COMFY_CONNECTION');

    process.env.OD_MEDIA_ALLOW_STUBS = '1';
    const audio = await generateMedia({
      projectRoot,
      projectsRoot,
      projectId: 'p1',
      surface: 'audio',
      audioKind: 'sfx',
      model: 'elevenlabs-sfx',
      prompt: 'short glass UI chime',
      output: 'chime.mp3',
      localOnly: true,
    }).catch((e: unknown) => e);
    expect(audio).not.toBeInstanceOf(MediaLocalOnlyError);
  });

  it('does nothing when the policy is off', async () => {
    const result = await generate('gpt-image-2', 'image', false).catch((e: unknown) => e);
    expect(result).not.toBeInstanceOf(MediaLocalOnlyError);
  });
});

describe('media catalogue readiness', () => {
  it('agent view, policy on, ComfyUI configured: only allowed models, comfyui-agent is the default', async () => {
    await writeMediaConfig({ comfyui: { baseUrl: 'http://127.0.0.1:8188', apiKey: 'aiand-key' } });
    const catalog = await buildMediaCatalog({ projectRoot, config: null, view: 'agent' });
    expect(catalog.policy).toMatchObject({ localOnly: true, view: 'agent', source: 'default' });
    expect(catalog.image.map((m) => m.id)).toEqual(['comfyui-agent', 'comfyui-workflow']);
    expect(catalog.image.filter((m) => m.default).map((m) => m.id)).toEqual(['comfyui-agent']);
    expect(catalog.image[0]).toMatchObject({ configured: true, allowed: true, ready: true });
    expect(catalog.video.map((m) => m.id)).toEqual(['hyperframes-html']);
    expect(catalog.video[0]).toMatchObject({ ready: true, default: true });
    expect(catalog.readiness.comfyui).toEqual({ configured: true, local: true, allowed: true });
    expect(catalog.readiness.vela).toMatchObject({ local: false, allowed: false });
    expect(catalog.readiness.elevenlabs).toMatchObject({ allowed: true });
  });

  it('agent view, policy on, ComfyUI not configured: no image default at all', async () => {
    const catalog = await buildMediaCatalog({ projectRoot, config: null, view: 'agent' });
    expect(catalog.image.every((m) => m.allowed)).toBe(true);
    expect(catalog.image.some((m) => m.default)).toBe(false);
    expect(catalog.image.find((m) => m.id === 'comfyui-agent')).toMatchObject({ configured: false, ready: false });
  });

  it('full view keeps hosted models, marked not allowed, and never marks them default', async () => {
    process.env.OPENAI_API_KEY = 'sk-test';
    const catalog = await buildMediaCatalog({ projectRoot, config: null, view: 'full' });
    const vela = catalog.image.find((m) => m.id === 'vela/gpt-image-2');
    expect(vela).toMatchObject({ allowed: false, ready: false });
    expect(vela?.default).toBeUndefined();
    expect(catalog.image.find((m) => m.id === 'gpt-image-2')).toMatchObject({ configured: true, allowed: false, ready: false });
    for (const list of [catalog.image, catalog.video, ...Object.values(catalog.audio)]) {
      expect(list.filter((m) => m.default).length).toBeLessThanOrEqual(1);
      for (const m of list.filter((x) => x.default)) expect(m.ready).toBe(true);
    }
  });

  it('policy off: a configured hosted model can be the default, an unconfigured one cannot', async () => {
    const off = await buildMediaCatalog({ projectRoot, config: { media: { localOnly: false } }, view: 'agent' });
    expect(off.image.some((m) => m.id === 'vela/gpt-image-2')).toBe(true);
    expect(off.image.find((m) => m.id === 'vela/gpt-image-2')?.default).toBeUndefined();
    process.env.OPENAI_API_KEY = 'sk-test';
    const withKey = await buildMediaCatalog({ projectRoot, config: { media: { localOnly: false } }, view: 'agent' });
    expect(withKey.image.filter((m) => m.default).map((m) => m.id)).toEqual(['gpt-image-2']);
  });
});
