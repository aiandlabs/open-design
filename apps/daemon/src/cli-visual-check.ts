// `od visual-check` — CLI half of POST /api/projects/:id/visual-check.
// cli.ts registers it in SUBCOMMAND_MAP and passes its shared helpers in,
// because cli.ts is the process entry and cannot be imported.

import {
  VISUAL_CHECK_DEFAULT_MODEL,
  visualCheckViewportsFromFlag,
  type VisualCheckRequest,
  type VisualCheckResponse,
} from '@open-design/contracts';

export const VISUAL_CHECK_STRING_FLAGS = new Set([
  'daemon-url', 'project', 'file', 'brief', 'prompt-file', 'checklist-file', 'viewport', 'model',
]);
export const VISUAL_CHECK_BOOLEAN_FLAGS = new Set(['help', 'h', 'json']);

export interface VisualCheckCliHelpers {
  parseFlags: (argv: string[], opts: { string: Set<string>; boolean: Set<string> }) => Record<string, any>;
  positionalArgs: (argv: string[], stringFlags: Set<string>) => string[];
  cliDaemonBaseUrl: (flags: Record<string, any>) => Promise<string>;
  surfaceFetchError: (err: unknown, base: string) => void;
  structuredHttpFailure: (resp: Response) => Promise<unknown>;
  /** Reads `--prompt-file <path|->` (stdin for `-`). */
  readTextFile: (pathOrDash: string) => Promise<string>;
  env?: NodeJS.ProcessEnv;
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  exit?: (code: number) => never;
}

export function printVisualCheckHelp(write: (text: string) => void = (t) => console.log(t)): void {
  write(`Usage:
  od visual-check <file> [--project <id>] [--brief <text> | --prompt-file <path|->]
                  [--checklist-file <path|->] [--viewport desktop|mobile|both]
                  [--model <id>] [--json]

Render an HTML file from the project to PNG and grade it with an ai& vision
model. Prints a verdict and, per requirement, pass/fail, the evidence, and a
repair hint. Screenshots are saved under the project's .visual-check/ folder.

Without --checklist-file the check grades a fixed baseline (clipped text,
contrast, overlap, horizontal overflow, broken images) plus requirements the
model derives from --brief / --prompt-file, or from the project's original
prompt when neither is given.

Rendering uses the desktop app's bundled Chromium when the desktop app is
running, otherwise a system Chrome / Edge / Chromium (override the binary
with OD_BROWSER_EXECUTABLE_PATH). The ai& key comes from Settings → Media
providers → ComfyUI or AIAND_API_KEY.

Options:
  --project <id>             Project id (default: $OD_PROJECT_ID)
  --brief <text>             What the page is meant to be
  --prompt-file <path|->     Read the brief from a file, or stdin with -
  --checklist-file <path|->  One requirement per line; replaces the baseline
  --viewport <v>             desktop (default) | mobile | both
  --model <id>               ai& vision model (default: Settings, then ${VISUAL_CHECK_DEFAULT_MODEL})
  --json                     Print the full result as JSON
  --daemon-url <url>         Override daemon URL

Exit codes: 0 graded (pass or fail; see the verdict) · 2 usage ·
3 daemon unreachable · other codes are structured daemon errors.

Examples:
  od visual-check index.html --brief "pricing page, three tiers, dark theme"
  od visual-check index.html --viewport both --json
  printf 'Hero headline fits on one line\\nCTA is visible above the fold\\n' | od visual-check index.html --checklist-file -`);
}

export function formatVisualCheckResult(result: VisualCheckResponse): string {
  const lines = [
    `verdict: ${result.verdict.toUpperCase()}  (${result.items.filter((i) => i.pass).length}/${result.items.length} passed)`,
  ];
  if (result.summary) lines.push(`summary: ${result.summary}`);
  lines.push('');
  for (const item of result.items) {
    lines.push(`[${item.pass ? 'PASS' : 'FAIL'}] ${item.id}: ${item.item}`);
    if (item.evidence) lines.push(`       evidence: ${item.evidence}${item.viewport ? ` (${item.viewport})` : ''}`);
    if (!item.pass && item.repairHint) lines.push(`       fix: ${item.repairHint}`);
  }
  lines.push('');
  for (const shot of result.screenshots) {
    lines.push(`screenshot (${shot.viewport} ${shot.width}x${shot.height}): ${shot.path}`);
  }
  const tokens = `${result.usage.inputTokens} in / ${result.usage.outputTokens} out`;
  const cost = result.estimatedCostUsd !== undefined ? ` · ~$${result.estimatedCostUsd.toFixed(4)}` : '';
  lines.push(`model: ${result.model} (${result.apiShape}) · tokens: ${tokens}${cost} · renderer: ${result.renderer} · ${(result.durationMs / 1000).toFixed(1)}s`);
  return lines.join('\n');
}

export async function runVisualCheckCommand(args: string[], h: VisualCheckCliHelpers): Promise<void> {
  const env = h.env ?? process.env;
  const out = h.stdout ?? ((t: string) => process.stdout.write(t));
  const err = h.stderr ?? ((t: string) => process.stderr.write(t));
  const exit = h.exit ?? ((code: number) => process.exit(code));
  if (args.length === 0 || args[0] === 'help' || args.includes('--help') || args.includes('-h')) {
    printVisualCheckHelp((t) => out(`${t}\n`));
    return exit(args.length === 0 ? 2 : 0);
  }
  let flags: Record<string, any>;
  try {
    flags = h.parseFlags(args, { string: VISUAL_CHECK_STRING_FLAGS, boolean: VISUAL_CHECK_BOOLEAN_FLAGS });
  } catch (e) {
    err(`${e instanceof Error ? e.message : String(e)}\n`);
    return exit(2);
  }
  const file = flags.file || h.positionalArgs(args, VISUAL_CHECK_STRING_FLAGS)[0];
  const projectId = flags.project || env.OD_PROJECT_ID;
  if (!file || !projectId) {
    err('od visual-check needs a <file> and --project <id> (or OD_PROJECT_ID)\n');
    return exit(2);
  }
  if (flags.brief && flags['prompt-file']) {
    err('use either --brief or --prompt-file, not both\n');
    return exit(2);
  }
  let viewports;
  try {
    viewports = visualCheckViewportsFromFlag(flags.viewport);
  } catch (e) {
    err(`${e instanceof Error ? e.message : String(e)}\n`);
    return exit(2);
  }
  const body: VisualCheckRequest = { fileName: String(file), viewports };
  try {
    const brief = flags['prompt-file'] ? await h.readTextFile(flags['prompt-file']) : flags.brief;
    if (typeof brief === 'string' && brief.trim()) body.brief = brief.trim();
    if (flags['checklist-file']) {
      const items = (await h.readTextFile(flags['checklist-file']))
        .split(/\r?\n/)
        .map((line) => line.replace(/^\s*(?:[-*]|\d+[.)])\s*/, '').trim())
        .filter(Boolean);
      if (items.length === 0) {
        err('--checklist-file has no requirements\n');
        return exit(2);
      }
      body.checklist = items;
    }
  } catch (e) {
    err(`${e instanceof Error ? e.message : String(e)}\n`);
    return exit(2);
  }
  if (typeof flags.model === 'string' && flags.model.trim()) body.model = flags.model.trim();

  const base = await h.cliDaemonBaseUrl(flags);
  const token = env.OD_TOOL_TOKEN;
  let resp: Response;
  try {
    resp = await fetch(`${base}/api/projects/${encodeURIComponent(projectId)}/visual-check`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify(body),
    });
  } catch (e) {
    h.surfaceFetchError(e, base);
    return exit(3);
  }
  if (!resp.ok) {
    await h.structuredHttpFailure(resp);
    return exit(1);
  }
  const result = (await resp.json()) as VisualCheckResponse;
  out(flags.json ? `${JSON.stringify(result, null, 2)}\n` : `${formatVisualCheckResult(result)}\n`);
}
