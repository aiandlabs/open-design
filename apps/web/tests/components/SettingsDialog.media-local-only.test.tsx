// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SettingsDialog } from '../../src/components/SettingsDialog';
import { mergeDaemonConfig } from '../../src/state/config';
import type { AppConfig } from '../../src/types';

const originalFetch = globalThis.fetch;

const baseConfig: AppConfig = {
  mode: 'api',
  apiKey: 'sk-test',
  apiProtocol: 'anthropic',
  baseUrl: 'https://api.anthropic.com',
  model: 'claude-sonnet-4-5',
  apiProviderBaseUrl: 'https://api.anthropic.com',
  agentId: null,
  skillId: null,
  designSystemId: null,
};

function renderMedia(initial: Partial<AppConfig> = {}) {
  const onPersist = vi.fn<(cfg: AppConfig) => void>();
  render(
    <SettingsDialog
      initial={{ ...baseConfig, ...initial }}
      agents={[]}
      daemonLive
      appVersionInfo={null}
      initialSection="media"
      onPersist={onPersist}
      onPersistComposioKey={vi.fn()}
      onClose={vi.fn()}
      onRefreshAgents={vi.fn()}
    />,
  );
  return { onPersist };
}

describe('Settings → Media providers → Local image generation only', () => {
  afterEach(() => {
    cleanup();
    vi.stubGlobal('fetch', originalFetch);
  });

  it('is on by default and marks hosted image providers as not used', () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 200 })));
    renderMedia();
    const toggle = screen.getByTestId('media-local-only-toggle');
    expect(toggle.getAttribute('role')).toBe('switch');
    expect(toggle.getAttribute('aria-checked')).toBe('true');
    expect(toggle.textContent).toContain('Local image generation only');

    fireEvent.click(screen.getByRole('tab', { name: /OpenAI/ }));
    expect(screen.getByTestId('media-local-only-blocked-note').textContent).toContain(
      'this provider is not used for images or video',
    );
    fireEvent.click(screen.getByRole('tab', { name: /ComfyUI/ }));
    expect(screen.queryByTestId('media-local-only-blocked-note')).toBeNull();
  });

  it('turns off and persists media.localOnly = false', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 200 })));
    const { onPersist } = renderMedia();
    fireEvent.click(screen.getByTestId('media-local-only-toggle'));
    expect(screen.getByTestId('media-local-only-toggle').getAttribute('aria-checked')).toBe('false');
    fireEvent.click(screen.getByRole('tab', { name: /OpenAI/ }));
    expect(screen.queryByTestId('media-local-only-blocked-note')).toBeNull();
    await waitFor(
      () => {
        const last = onPersist.mock.calls.at(-1)?.[0];
        expect(last?.media).toEqual({ localOnly: false });
      },
      { timeout: 4000 },
    );
  });

  it('shows a stored off value', () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 200 })));
    renderMedia({ media: { localOnly: false } });
    expect(screen.getByTestId('media-local-only-toggle').getAttribute('aria-checked')).toBe('false');
  });

  it('takes the daemon value on config merge', () => {
    expect(mergeDaemonConfig(baseConfig, { media: { localOnly: false } }).media).toEqual({ localOnly: false });
    expect(mergeDaemonConfig({ ...baseConfig, media: { localOnly: false } }, { media: null }).media).toBeUndefined();
  });
});
