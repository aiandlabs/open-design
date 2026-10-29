// Grade rendered screenshots against a checklist with an ai& vision model.
//
// The call goes through `AiandClient` from @open-design/comfy-agent (the same
// client the ComfyUI asset loop's critic uses), with structured output forced
// through a tool call. The model gets the screenshots, the checklist, and —
// when no explicit checklist was given — the brief, from which it adds a few
// brief-specific requirements to the fixed baseline below.

import type { AiandClient, ApiShape, ContentPart, JsonSchema } from '@open-design/comfy-agent';
import type {
  VisualCheckChecklistSource,
  VisualCheckItemResult,
  VisualCheckUsage,
  VisualCheckViewportName,
} from '@open-design/contracts';

/** Always graded unless the caller supplies an explicit checklist. */
export const VISUAL_CHECK_BASELINE_CHECKLIST: readonly string[] = [
  'All text is fully visible: no text is cut off, clipped, hidden behind other elements, or running past the edge of its container or the page.',
  'Every piece of text has enough contrast against its background to read easily (at least WCAG AA, about 4.5:1 for body text).',
  'No elements overlap or collide in a way that hides content or makes it hard to read.',
  'The layout fits the viewport width: nothing is cut off at the left or right edge and there is no unintended horizontal overflow.',
  'There are no broken images, empty placeholder boxes, or visibly unstyled content.',
];

const MAX_DERIVED_ITEMS = 6;

export interface VisualCheckGradeInput {
  images: Array<{ viewport: VisualCheckViewportName; png: Buffer; width: number; height: number }>;
  checklist?: string[] | undefined;
  brief?: string | undefined;
  briefSource?: 'brief' | 'project-prompt' | undefined;
  /** BCP 47 language for everything the grader writes; English when absent. */
  language?: string | undefined;
  model: string;
}

export interface VisualCheckGradeResult {
  summary: string;
  items: VisualCheckItemResult[];
  checklistSource: VisualCheckChecklistSource;
  model: string;
  apiShape: ApiShape;
  usage: VisualCheckUsage;
  estimatedCostUsd?: number;
}

type RawItem = {
  id?: unknown;
  item?: unknown;
  pass?: unknown;
  evidence?: unknown;
  repairHint?: unknown;
  viewport?: unknown;
};

const SYSTEM_PROMPT = `You are a strict visual QA reviewer for web pages. You receive full-page screenshots of one HTML file and a list of requirements. Grade each requirement as pass or fail using only what is visible in the screenshots.

Rules:
- Fail a requirement when any visible part of the page breaks it. One clipped heading or one unreadable label is enough to fail.
- Evidence must name the specific element, its text, and where it is (for example "the hero heading 'Annual report' is cut off at the right edge of the desktop screenshot").
- For a failing item, the repair hint must be one concrete change to the HTML/CSS (for example "allow the heading to wrap: remove white-space: nowrap and set overflow-wrap: anywhere"). For a passing item, leave the repair hint empty.
- Do not invent problems you cannot see. Do not grade code quality or anything not visible.
- Set viewport to the screenshot the evidence comes from when it is specific to one viewport.`;

function buildSchema(viewports: readonly VisualCheckViewportName[]): JsonSchema {
  return {
    type: 'object',
    additionalProperties: false,
    required: ['summary', 'items'],
    properties: {
      summary: { type: 'string', description: 'One or two sentences on the overall state of the page.' },
      items: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['id', 'item', 'pass', 'evidence', 'repairHint'],
          properties: {
            id: { type: 'string', description: 'The requirement id as given, or d1, d2, ... for requirements you derived from the brief.' },
            item: { type: 'string', description: 'The requirement text.' },
            pass: { type: 'boolean' },
            evidence: { type: 'string' },
            repairHint: { type: 'string' },
            viewport: { type: 'string', enum: [...viewports] },
          },
        },
      },
    },
  };
}

export function resolveChecklist(input: Pick<VisualCheckGradeInput, 'checklist' | 'brief' | 'briefSource'>): {
  given: Array<{ id: string; item: string }>;
  source: VisualCheckChecklistSource;
  deriveFromBrief: boolean;
} {
  if (input.checklist && input.checklist.length > 0) {
    return {
      given: input.checklist.map((item, i) => ({ id: `c${i + 1}`, item })),
      source: 'request',
      deriveFromBrief: false,
    };
  }
  const given = VISUAL_CHECK_BASELINE_CHECKLIST.map((item, i) => ({ id: `b${i + 1}`, item }));
  if (input.brief?.trim()) {
    return { given, source: input.briefSource ?? 'brief', deriveFromBrief: true };
  }
  return { given, source: 'default', deriveFromBrief: false };
}

/** English name of a BCP 47 language, for the prompt ("ja" → "Japanese"). */
function languageName(tag: string): string {
  try {
    return new Intl.DisplayNames(['en'], { type: 'language' }).of(tag) ?? tag;
  } catch {
    return tag;
  }
}

export function buildGradeRequestText(
  input: Pick<VisualCheckGradeInput, 'checklist' | 'brief' | 'briefSource' | 'images' | 'language'>,
): string {
  const { given, deriveFromBrief } = resolveChecklist(input);
  const shots = input.images
    .map((img, i) => `Screenshot ${i + 1}: ${img.viewport} viewport, ${img.width}x${img.height} CSS px.`)
    .join('\n');
  const lines = [
    shots,
    '',
    'Requirements to grade (return one item per requirement, in this order, with the same id):',
    ...given.map((g) => `- ${g.id}: ${g.item}`),
  ];
  if (deriveFromBrief) {
    lines.push(
      '',
      `Brief for this page:\n"""\n${input.brief!.trim()}\n"""`,
      '',
      `Also derive up to ${MAX_DERIVED_ITEMS} visual requirements that this brief makes checkable in a screenshot (required sections, content, hierarchy, mood, palette). Give them ids d1, d2, ... and grade them after the requirements above. Skip anything already covered above or not visible in a screenshot.`,
    );
  }
  if (input.language) {
    lines.push(
      '',
      `Write the summary, every item text, evidence and repairHint in ${languageName(input.language)} (${input.language}). Translate the requirement texts above into it, and keep their ids.`,
    );
  }
  return lines.join('\n');
}

function cleanString(value: unknown, max = 1200): string {
  return typeof value === 'string' ? value.trim().slice(0, max) : '';
}

/** Normalize the model's items; given requirements the model skipped become failures. */
export function normalizeGradedItems(
  raw: unknown,
  given: Array<{ id: string; item: string }>,
  viewports: readonly VisualCheckViewportName[],
  allowDerived: boolean,
  translated = false,
): VisualCheckItemResult[] {
  const list: RawItem[] = Array.isArray(raw) ? (raw as RawItem[]) : [];
  const byId = new Map<string, RawItem>();
  for (const entry of list) {
    if (!entry || typeof entry !== 'object') continue;
    const id = cleanString(entry.id, 20);
    if (id && !byId.has(id)) byId.set(id, entry);
  }
  const toResult = (id: string, item: string, entry: RawItem | undefined): VisualCheckItemResult => {
    if (!entry) {
      return {
        id,
        item,
        pass: false,
        evidence: 'The model returned no grade for this requirement.',
        repairHint: 'Run the check again.',
      };
    }
    const pass = entry.pass === true;
    const viewport = typeof entry.viewport === 'string' && (viewports as readonly string[]).includes(entry.viewport)
      ? (entry.viewport as VisualCheckViewportName)
      : undefined;
    return {
      id,
      item,
      pass,
      evidence: cleanString(entry.evidence),
      repairHint: pass ? '' : cleanString(entry.repairHint),
      ...(viewport ? { viewport } : {}),
    };
  };
  // With a requested language the model's own item text is the translation; else keep the given text.
  const results = given.map((g) => {
    const entry = byId.get(g.id);
    const text = translated ? cleanString(entry?.item, 300) || g.item : g.item;
    return toResult(g.id, text, entry);
  });
  if (allowDerived) {
    let count = 0;
    for (const [id, entry] of byId) {
      if (!/^d\d+$/.test(id) || count >= MAX_DERIVED_ITEMS) continue;
      const item = cleanString(entry.item, 300);
      if (!item) continue;
      results.push(toResult(id, item, entry));
      count += 1;
    }
  }
  return results;
}

export async function gradeVisualCheck(
  client: Pick<AiandClient, 'structured'>,
  input: VisualCheckGradeInput,
): Promise<VisualCheckGradeResult> {
  const { given, source, deriveFromBrief } = resolveChecklist(input);
  const viewports = input.images.map((img) => img.viewport);
  const content: ContentPart[] = [];
  for (const img of input.images) {
    content.push({ type: 'image', mime: 'image/png', base64: img.png.toString('base64') });
  }
  content.push({ type: 'text', text: buildGradeRequestText(input) });
  const result = await client.structured<{ summary?: unknown; items?: unknown }>({
    model: input.model,
    system: SYSTEM_PROMPT,
    content,
    schema: buildSchema(viewports),
    toolName: 'report_visual_check',
    toolDescription: 'Report the pass/fail grade, evidence, and repair hint for every requirement.',
    maxTokens: 6000,
    reasoningEffort: 'low',
    temperature: 0,
  });
  const items = normalizeGradedItems(result.data?.items, given, viewports, deriveFromBrief, Boolean(input.language));
  return {
    summary: cleanString(result.data?.summary, 600),
    items,
    checklistSource: source,
    model: result.model,
    apiShape: result.shape,
    usage: { ...result.usage },
    ...(result.estimatedCostUsd !== undefined ? { estimatedCostUsd: result.estimatedCostUsd } : {}),
  };
}
