// @vitest-environment jsdom

import type { GeneratedAsset } from '@open-design/contracts';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GeneratedAssetReview } from '../../src/components/generated-assets/GeneratedAssetReview';

const originalFetch = globalThis.fetch;

const leaf: GeneratedAsset = {
  path: 'leaf.png',
  tracePath: 'leaf.trace.json',
  model: 'comfyui-agent',
  kind: 'icon',
  set: 'onboarding',
  exists: true,
  review: { state: 'accepted', reason: 'default' },
};

describe('GeneratedAssetReview', () => {
  afterEach(() => {
    cleanup();
    vi.stubGlobal('fetch', originalFetch);
  });

  it('shows the review state of a generated image and rejects it through the review endpoint', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === 'PUT') {
        return new Response(JSON.stringify({ asset: { ...leaf, review: { state: 'rejected', reason: 'user', via: 'ui' } } }), { status: 200 });
      }
      return new Response(JSON.stringify({ assets: [leaf] }), { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<GeneratedAssetReview projectId="proj 1" fileName="leaf.png" />);

    expect(await screen.findByText('Accepted')).toBeTruthy();
    expect(fetchMock.mock.calls[0]![0]).toBe('/api/projects/proj%201/media/assets');
    fireEvent.click(screen.getByRole('button', { name: 'Reject' }));

    expect(await screen.findByText('Rejected')).toBeTruthy();
    const [url, init] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
    expect(url).toBe('/api/projects/proj%201/media/assets/review');
    expect(init.method).toBe('PUT');
    expect(JSON.parse(String(init.body))).toEqual({ path: 'leaf.png', state: 'rejected', via: 'ui' });
    expect(screen.getByRole('button', { name: 'Accept' })).toBeTruthy();
  });

  it('renders nothing for a file the ComfyUI provider did not make', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ assets: [leaf] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const { container } = render(<GeneratedAssetReview projectId="p" fileName="photo.png" />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(container.textContent).toBe('');
  });

  it('keeps the previous state and says so when the review fails', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) =>
      init?.method === 'PUT'
        ? new Response(JSON.stringify({ error: { message: 'disk full' } }), { status: 500 })
        : new Response(JSON.stringify({ assets: [leaf] }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    render(<GeneratedAssetReview projectId="p" fileName="leaf.png" />);
    fireEvent.click(await screen.findByRole('button', { name: 'Reject' }));
    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(screen.getByText('Accepted')).toBeTruthy();
  });
});
