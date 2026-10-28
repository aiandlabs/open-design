import type {
  GeneratedAsset,
  GeneratedAssetReviewResponse,
  GeneratedAssetReviewState,
  GeneratedAssetsResponse,
} from '@open-design/contracts';

function assetsBase(projectId: string): string {
  return `/api/projects/${encodeURIComponent(projectId)}/media/assets`;
}

async function readJson<T>(resp: Response, what: string): Promise<T> {
  const payload = (await resp.json().catch(() => null)) as (T & { error?: { message?: string } }) | null;
  if (!resp.ok || !payload) throw new Error(payload?.error?.message?.trim() || `${what} failed (${resp.status})`);
  return payload;
}

/** GET /api/projects/:id/media/assets — the same endpoint `od media assets` calls. */
export async function listGeneratedAssets(projectId: string, init: { signal?: AbortSignal } = {}): Promise<GeneratedAsset[]> {
  const resp = await fetch(assetsBase(projectId), init.signal ? { signal: init.signal } : {});
  return (await readJson<GeneratedAssetsResponse>(resp, 'Listing generated assets')).assets;
}

/** PUT /api/projects/:id/media/assets/review — the same endpoint `od media review` calls. */
export async function reviewGeneratedAsset(projectId: string, path: string, state: GeneratedAssetReviewState): Promise<GeneratedAsset> {
  const resp = await fetch(`${assetsBase(projectId)}/review`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path, state, via: 'ui' }),
  });
  return (await readJson<GeneratedAssetReviewResponse>(resp, 'Reviewing the asset')).asset;
}
