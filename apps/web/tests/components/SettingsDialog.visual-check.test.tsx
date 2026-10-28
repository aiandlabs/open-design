// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SettingsDialog } from '../../src/components/SettingsDialog';
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

function renderInstructions(initial: Partial<AppConfig> = {}) {
  const onPersist = vi.fn<(cfg: AppConfig) => void>();
  render(
    <SettingsDialog
      initial={{ ...baseConfig, ...initial }}
      agents={[]}
      daemonLive
      appVersionInfo={null}
      initialSection="instructions"
      onPersist={onPersist}
      onPersistComposioKey={vi.fn()}
      onClose={vi.fn()}
      onRefreshAgents={vi.fn()}
    />,
  );
  return { onPersist };
}

describe('Settings → Instructions → Visual check', () => {
  afterEach(() => {
    cleanup();
    vi.stubGlobal('fetch', originalFetch);
  });

  it('defaults to off, with automatic model selection as the placeholder', () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 200 })));
    renderInstructions();
    const rounds = screen.getByTestId('visual-check-rounds') as HTMLSelectElement;
    expect(rounds.value).toBe('0');
    expect(Array.from(rounds.options).map((o) => o.value)).toEqual(['0', '1', '2', '3']);
    expect(rounds.options[0]!.textContent).toBe('Off');
    const model = screen.getByTestId('visual-check-model') as HTMLInputElement;
    expect(model.value).toBe('');
    // No model id is named: an empty field means the daemon picks one from the ai& list.
    expect(model.placeholder).toBe('Automatic: a vision model from ai&');
  });

  it('shows the stored value and persists rounds and model changes', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 200 })));
    const { onPersist } = renderInstructions({ visualCheck: { rounds: 1 } });
    const rounds = screen.getByTestId('visual-check-rounds') as HTMLSelectElement;
    expect(rounds.value).toBe('1');

    fireEvent.change(rounds, { target: { value: '2' } });
    fireEvent.change(screen.getByTestId('visual-check-model'), { target: { value: 'vendor/vision' } });

    await waitFor(
      () => {
        const last = onPersist.mock.calls.at(-1)?.[0];
        expect(last?.visualCheck).toEqual({ rounds: 2, model: 'vendor/vision' });
      },
      { timeout: 4000 },
    );
  });
});
