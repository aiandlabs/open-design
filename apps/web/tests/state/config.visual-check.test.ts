import { afterEach, describe, expect, it, vi } from 'vitest';

import { DEFAULT_CONFIG, mergeDaemonConfig, syncConfigToDaemon } from '../../src/state/config';

const originalFetch = globalThis.fetch;

describe('visual check settings <-> daemon app-config', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal('fetch', originalFetch);
  });

  it('reads visualCheck from the daemon and clamps rounds to the max', () => {
    const merged = mergeDaemonConfig(DEFAULT_CONFIG, {
      visualCheck: { rounds: 9, model: '  zai-org/glm-5.3-flash ' },
    });
    expect(merged.visualCheck).toEqual({ rounds: 3, model: 'zai-org/glm-5.3-flash' });
  });

  it('keeps the local value when the daemon has no visualCheck key', () => {
    const merged = mergeDaemonConfig({ ...DEFAULT_CONFIG, visualCheck: { rounds: 2 } }, {});
    expect(merged.visualCheck).toEqual({ rounds: 2 });
  });

  it('writes visualCheck through the app-config PUT', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    await syncConfigToDaemon({ ...DEFAULT_CONFIG, visualCheck: { rounds: 2, model: 'vendor/vision' } });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/app-config');
    expect(JSON.parse(String(init.body)).visualCheck).toEqual({ rounds: 2, model: 'vendor/vision' });
  });

  it('omits visualCheck when the user never touched it, so the daemon default (off) holds', async () => {
    const fetchMock = vi.fn(async () => new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    await syncConfigToDaemon({ ...DEFAULT_CONFIG });
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(String(init.body))).not.toHaveProperty('visualCheck');
  });
});
