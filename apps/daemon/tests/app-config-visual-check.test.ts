import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { readAppConfig, writeAppConfig } from '../src/app-config.js';

let dataDir = '';

beforeEach(async () => {
  dataDir = await mkdtemp(path.join(tmpdir(), 'od-app-config-visual-check-'));
});

afterEach(async () => {
  await rm(dataDir, { recursive: true, force: true });
});

describe('app-config visualCheck', () => {
  it('is absent by default, so rounds resolve to 0 (off)', async () => {
    expect((await readAppConfig(dataDir)).visualCheck).toBeUndefined();
  });

  it('persists rounds and model, clamping rounds to the 0..3 range', async () => {
    await writeAppConfig(dataDir, { visualCheck: { rounds: 2, model: ' some/vision ' } });
    expect((await readAppConfig(dataDir)).visualCheck).toEqual({ rounds: 2, model: 'some/vision' });
    await writeAppConfig(dataDir, { visualCheck: { rounds: 12 } });
    expect((await readAppConfig(dataDir)).visualCheck).toEqual({ rounds: 3 });
    await writeAppConfig(dataDir, { visualCheck: { rounds: -1, model: '' } });
    expect((await readAppConfig(dataDir)).visualCheck).toEqual({ rounds: 0 });
  });

  it('drops malformed values and keeps other keys', async () => {
    await writeAppConfig(dataDir, { customInstructions: 'keep me', visualCheck: 'three' });
    const cfg = await readAppConfig(dataDir);
    expect(cfg.visualCheck).toBeUndefined();
    expect(cfg.customInstructions).toBe('keep me');
  });
});
