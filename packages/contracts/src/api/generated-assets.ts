// Images the local ComfyUI provider generated in a project, and the user's
// review of each one. A generated asset is accepted unless the user rejects
// it or deletes its file. Accepted icons from the same project (and set) are
// the style references for the next icon, so a rejection keeps a bad icon
// from steering later ones.

export const GENERATED_ASSET_REVIEW_STATES = ['accepted', 'rejected'] as const;
export type GeneratedAssetReviewState = (typeof GENERATED_ASSET_REVIEW_STATES)[number];

/** Why an asset has its state: never reviewed, set by a person, or its file is gone. */
export type GeneratedAssetReviewReason = 'default' | 'user' | 'deleted';

/** Which surface set the state. */
export type GeneratedAssetReviewVia = 'ui' | 'cli' | 'api';

export interface GeneratedAssetReview {
  state: GeneratedAssetReviewState;
  reason: GeneratedAssetReviewReason;
  updatedAt?: string;
  via?: GeneratedAssetReviewVia;
}

export interface GeneratedAsset {
  /** Project-relative path of the image. */
  path: string;
  /** Project-relative path of its `<name>.trace.json`, which also stores the review. */
  tracePath: string;
  model: 'comfyui-agent' | 'comfyui-workflow';
  /** `icon` when the asset was generated as an icon. */
  kind?: 'icon';
  /** Icon set name the generation was asked to join. */
  set?: string;
  prompt?: string;
  createdAt?: string;
  /** False when the image file was deleted; such an asset counts as rejected. */
  exists: boolean;
  review: GeneratedAssetReview;
}

export interface GeneratedAssetsResponse {
  assets: GeneratedAsset[];
  /** Asset loops still running (their trace is rewritten after each round). */
  generating?: GeneratingAsset[];
}

/** An asset loop that has not finished: its work folder and progress trace. */
export interface GeneratingAsset {
  /** Project-relative work folder, `.comfy-agent/<name>`, holding round-N.png. */
  workDir: string;
  /** Project-relative path of the progress trace. */
  tracePath: string;
  prompt?: string;
  kind?: 'icon';
  set?: string;
  startedAt?: string;
}

export interface GeneratedAssetReviewRequest {
  /** Project-relative path of the image. */
  path: string;
  state: GeneratedAssetReviewState;
  via?: GeneratedAssetReviewVia;
}

export interface GeneratedAssetReviewResponse {
  asset: GeneratedAsset;
}

/** Strict parser for {@link GeneratedAssetReviewRequest}; returns an error message instead of throwing. */
export function parseGeneratedAssetReviewRequest(body: unknown): { ok: true; value: GeneratedAssetReviewRequest } | { ok: false; error: string } {
  if (!body || typeof body !== 'object') return { ok: false, error: 'body must be an object' };
  const b = body as Record<string, unknown>;
  if (typeof b.path !== 'string' || !b.path.trim()) return { ok: false, error: 'path must be a project-relative file path' };
  if (!GENERATED_ASSET_REVIEW_STATES.includes(b.state as GeneratedAssetReviewState)) {
    return { ok: false, error: `state must be one of ${GENERATED_ASSET_REVIEW_STATES.join(', ')}` };
  }
  const via = b.via === 'ui' || b.via === 'cli' || b.via === 'api' ? b.via : undefined;
  return { ok: true, value: { path: b.path.trim(), state: b.state as GeneratedAssetReviewState, ...(via ? { via } : {}) } };
}
