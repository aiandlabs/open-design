// @vitest-environment jsdom

import type { VisualCheckResponse } from '@open-design/contracts';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { VisualCheckAction } from '../../src/components/visual-check/VisualCheckAction';

const originalFetch = globalThis.fetch;

const response: VisualCheckResponse = {
  verdict: 'fail',
  summary: 'The hero headline overflows its card.',
  items: [
    {
      id: 'c1',
      item: 'Headline fits inside the hero card',
      pass: false,
      evidence: 'The headline runs past the right edge of the card.',
      repairHint: 'Allow the headline to wrap or reduce its font size.',
      viewport: 'desktop',
    },
    {
      id: 'c2',
      item: 'Body text has enough contrast',
      pass: true,
      evidence: 'Dark text on a white background.',
      repairHint: '',
    },
  ],
  checklistSource: 'brief',
  screenshots: [
    { viewport: 'desktop', width: 1440, height: 900, path: '.visual-check/index/1-desktop.png', bytes: 1234 },
  ],
  renderer: 'system-browser',
  model: 'zai-org/glm-5.3-flash',
  apiShape: 'openai',
  usage: { inputTokens: 2100, outputTokens: 340 },
  durationMs: 4200,
};

describe('VisualCheckAction', () => {
  afterEach(() => {
    cleanup();
    vi.stubGlobal('fetch', originalFetch);
  });

  it('posts the file, viewports and brief to the visual-check endpoint and renders the grades', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(response), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    render(<VisualCheckAction projectId="proj 1" fileName="pages/index.html" />);

    fireEvent.click(screen.getByRole('button', { name: 'Check with vision model' }));
    fireEvent.change(screen.getByTestId('visual-check-viewport'), { target: { value: 'both' } });
    fireEvent.change(screen.getByTestId('visual-check-brief'), { target: { value: '  A landing page  ' } });
    fireEvent.click(screen.getByTestId('visual-check-run'));

    await screen.findByTestId('visual-check-result');
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/projects/proj%201/visual-check');
    expect(init.method).toBe('POST');
    expect(JSON.parse(String(init.body))).toEqual({
      fileName: 'pages/index.html',
      viewports: ['desktop', 'mobile'],
      brief: 'A landing page',
    });

    expect(screen.getByTestId('visual-check-verdict').textContent).toBe('Fail');
    expect(screen.getByText('The hero headline overflows its card.')).toBeTruthy();
    const items = screen.getAllByTestId('visual-check-item');
    expect(items.map((el) => el.getAttribute('data-pass'))).toEqual(['false', 'true']);
    expect(items[0]!.textContent).toContain('Allow the headline to wrap or reduce its font size.');
    // Passing items do not show a repair hint row.
    expect(items[1]!.textContent).not.toContain('Repair hint');
    const img = screen.getByRole('img', { name: 'Rendered desktop view' }) as HTMLImageElement;
    expect(img.getAttribute('src')).toBe(
      '/api/projects/proj%201/visual-check/screenshot?path=.visual-check%2Findex%2F1-desktop.png',
    );
    expect(screen.getByTestId('visual-check-usage').textContent).toBe(
      'zai-org/glm-5.3-flash · 2100 input / 340 output tokens',
    );
  });

  it('sends only the desktop viewport and no brief by default', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ...response, verdict: 'pass', items: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    render(<VisualCheckAction projectId="p" fileName="index.html" />);
    fireEvent.click(screen.getByTestId('visual-check-button'));
    fireEvent.click(screen.getByTestId('visual-check-run'));
    await waitFor(() => expect(screen.getByTestId('visual-check-verdict').textContent).toBe('Pass'));
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({ fileName: 'index.html', viewports: ['desktop'] });
  });

  it('shows the daemon error message when the check fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ error: { code: 'VISUAL_CHECK_NO_KEY', message: 'no ai& API key' } }), { status: 400 })),
    );
    render(<VisualCheckAction projectId="p" fileName="index.html" />);
    fireEvent.click(screen.getByTestId('visual-check-button'));
    fireEvent.click(screen.getByTestId('visual-check-run'));
    const alert = await screen.findByTestId('visual-check-error');
    expect(alert.textContent).toBe('Visual check failed: no ai& API key');
  });

  it('is disabled for read-only viewers', () => {
    render(<VisualCheckAction projectId="p" fileName="index.html" disabled disabledTitle="Read only" />);
    expect((screen.getByTestId('visual-check-button') as HTMLButtonElement).disabled).toBe(true);
  });
});
