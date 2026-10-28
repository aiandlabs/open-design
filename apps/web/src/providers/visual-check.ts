import type { VisualCheckRequest, VisualCheckResponse } from '@open-design/contracts';

function projectBase(projectId: string): string {
  return `/api/projects/${encodeURIComponent(projectId)}/visual-check`;
}

/** POST /api/projects/:id/visual-check — the same endpoint `od visual-check` calls. */
export async function runVisualCheck(
  projectId: string,
  request: VisualCheckRequest,
  init: { signal?: AbortSignal } = {},
): Promise<VisualCheckResponse> {
  const resp = await fetch(projectBase(projectId), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(request),
    ...(init.signal ? { signal: init.signal } : {}),
  });
  const payload = (await resp.json().catch(() => null)) as
    | (VisualCheckResponse & { error?: { message?: string } })
    | null;
  if (!resp.ok || !payload) {
    throw new Error(payload?.error?.message?.trim() || `Visual check failed (${resp.status})`);
  }
  return payload;
}

/** URL of one screenshot the check saved under the project's `.visual-check/` folder. */
export function visualCheckScreenshotUrl(projectId: string, path: string): string {
  return `${projectBase(projectId)}/screenshot?path=${encodeURIComponent(path)}`;
}
