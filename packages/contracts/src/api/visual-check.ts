// Visual check: render one project file to PNG and grade the picture with a
// vision model. Shared by `POST /api/projects/:id/visual-check`, the
// `od visual-check` CLI, and the file preview's "Check with vision model"
// action. The project travels in the URL path; everything else is the body.

export const VISUAL_CHECK_VIEWPORTS = ['desktop', 'mobile'] as const;
export type VisualCheckViewportName = (typeof VISUAL_CHECK_VIEWPORTS)[number];

/** CSS-pixel viewport each named preset renders at. */
export const VISUAL_CHECK_VIEWPORT_SIZES: Readonly<
  Record<VisualCheckViewportName, { width: number; height: number }>
> = {
  desktop: { width: 1440, height: 900 },
  mobile: { width: 390, height: 844 },
};

/** Upper bound for the per-task rounds setting. 0 means the loop is off. */
export const VISUAL_CHECK_MAX_ROUNDS = 3;
export const VISUAL_CHECK_MAX_CHECKLIST_ITEMS = 20;
export const VISUAL_CHECK_MAX_ITEM_CHARS = 300;
export const VISUAL_CHECK_MAX_BRIEF_CHARS = 8000;
export const VISUAL_CHECK_MAX_MODEL_CHARS = 200;

export interface VisualCheckRequest {
  /** Project-relative path of the HTML file to render. */
  fileName: string;
  /** What the design is meant to be. Used to derive a checklist when none is given. */
  brief?: string;
  /** Explicit pass/fail items. Takes precedence over `brief`. */
  checklist?: string[];
  /** Defaults to `['desktop']`. */
  viewports?: VisualCheckViewportName[];
  /**
   * ai& model id. Defaults to the Settings value; with neither, the daemon
   * picks a model with vision and tool calling from the live ai& model list.
   */
  model?: string;
}

/** Where the checklist the model graded against came from. */
export type VisualCheckChecklistSource = 'request' | 'brief' | 'project-prompt' | 'default';

/** Which renderer produced the PNGs. */
export type VisualCheckRenderer = 'desktop' | 'system-browser';

export interface VisualCheckScreenshot {
  viewport: VisualCheckViewportName;
  width: number;
  height: number;
  /** Project-relative path, inside the project's `.visual-check/` folder. */
  path: string;
  bytes: number;
}

export interface VisualCheckItemResult {
  id: string;
  /** The requirement as graded. */
  item: string;
  pass: boolean;
  /** What the model saw in the screenshot(s) that decided the result. */
  evidence: string;
  /** Concrete change that would make a failing item pass. Empty for passes. */
  repairHint: string;
  /** The viewport the evidence comes from, when it is specific to one. */
  viewport?: VisualCheckViewportName;
}

export interface VisualCheckUsage {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens?: number;
  reasoningTokens?: number;
}

export interface VisualCheckResponse {
  verdict: 'pass' | 'fail';
  /** One or two sentences from the model. */
  summary: string;
  items: VisualCheckItemResult[];
  checklistSource: VisualCheckChecklistSource;
  screenshots: VisualCheckScreenshot[];
  renderer: VisualCheckRenderer;
  model: string;
  apiShape: 'openai' | 'anthropic';
  usage: VisualCheckUsage;
  /** Cost estimate from the API's published per-token prices, when known. */
  estimatedCostUsd?: number;
  durationMs: number;
}

/** Settings → visual check. Stored in app-config as `visualCheck`. */
export interface VisualCheckPrefs {
  /** Checks the agent may run per task. 0 (default) turns the loop off. */
  rounds?: number;
  /** ai& vision model id. Absent means the daemon picks one from the ai& model list. */
  model?: string | null;
}

/** Clamp any stored or requested rounds value to 0..VISUAL_CHECK_MAX_ROUNDS. */
export function normalizeVisualCheckRounds(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 0;
  return Math.min(VISUAL_CHECK_MAX_ROUNDS, Math.max(0, Math.trunc(value)));
}

/** Normalize stored prefs; returns undefined when nothing meaningful remains. */
export function normalizeVisualCheckPrefs(value: unknown): VisualCheckPrefs | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const raw = value as Record<string, unknown>;
  const out: VisualCheckPrefs = { rounds: normalizeVisualCheckRounds(raw.rounds) };
  if (typeof raw.model === 'string') {
    const model = raw.model.trim().slice(0, VISUAL_CHECK_MAX_MODEL_CHARS);
    if (model) out.model = model;
  }
  return out;
}

function isViewportName(value: unknown): value is VisualCheckViewportName {
  return typeof value === 'string' && (VISUAL_CHECK_VIEWPORTS as readonly string[]).includes(value);
}

/**
 * Parse a request body. Rejects unknown fields so a caller cannot smuggle
 * credentials, a different project, or an output path.
 */
export function parseVisualCheckRequest(input: unknown): VisualCheckRequest {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new Error('visual check request must be an object');
  }
  const body = input as Record<string, unknown>;
  const allowed = new Set(['fileName', 'brief', 'checklist', 'viewports', 'model']);
  for (const key of Object.keys(body)) {
    if (!allowed.has(key)) throw new Error(`unsupported visual check request field: ${key}`);
  }
  if (typeof body.fileName !== 'string' || !body.fileName.trim()) {
    throw new Error('fileName is required');
  }
  const out: VisualCheckRequest = { fileName: body.fileName.trim() };
  if (body.brief !== undefined) {
    if (typeof body.brief !== 'string') throw new Error('brief must be a string');
    const brief = body.brief.trim();
    if (brief.length > VISUAL_CHECK_MAX_BRIEF_CHARS) {
      throw new Error(`brief must be at most ${VISUAL_CHECK_MAX_BRIEF_CHARS} characters`);
    }
    if (brief) out.brief = brief;
  }
  if (body.checklist !== undefined) {
    if (!Array.isArray(body.checklist)) throw new Error('checklist must be an array of strings');
    const items = body.checklist.map((item) => {
      if (typeof item !== 'string') throw new Error('checklist must be an array of strings');
      const text = item.trim();
      if (text.length > VISUAL_CHECK_MAX_ITEM_CHARS) {
        throw new Error(`checklist items must be at most ${VISUAL_CHECK_MAX_ITEM_CHARS} characters`);
      }
      return text;
    }).filter((item) => item.length > 0);
    if (items.length > VISUAL_CHECK_MAX_CHECKLIST_ITEMS) {
      throw new Error(`checklist must have at most ${VISUAL_CHECK_MAX_CHECKLIST_ITEMS} items`);
    }
    if (items.length > 0) out.checklist = items;
  }
  if (body.viewports !== undefined) {
    if (!Array.isArray(body.viewports) || body.viewports.length === 0 || !body.viewports.every(isViewportName)) {
      throw new Error(`viewports must be a non-empty array of: ${VISUAL_CHECK_VIEWPORTS.join(', ')}`);
    }
    out.viewports = VISUAL_CHECK_VIEWPORTS.filter((name) => (body.viewports as unknown[]).includes(name));
  }
  if (body.model !== undefined) {
    if (typeof body.model !== 'string' || !body.model.trim() || body.model.length > VISUAL_CHECK_MAX_MODEL_CHARS) {
      throw new Error('model must be a non-empty model id');
    }
    out.model = body.model.trim();
  }
  return out;
}

/** CLI `--viewport desktop|mobile|both` → request viewports. */
export function visualCheckViewportsFromFlag(value: string | undefined): VisualCheckViewportName[] {
  if (value === undefined || value === '' || value === 'desktop') return ['desktop'];
  if (value === 'mobile') return ['mobile'];
  if (value === 'both') return ['desktop', 'mobile'];
  throw new Error(`--viewport must be desktop, mobile, or both (got ${value})`);
}
