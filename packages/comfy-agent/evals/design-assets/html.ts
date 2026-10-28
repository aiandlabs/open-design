// Self-contained HTML contact sheets (images inlined as data URLs, no
// external assets), so they open offline and can be passed around as one file.

import { readFile } from "node:fs/promises";
import path from "node:path";

import { decodePng, encodePng, fitWithin } from "../../src/image.js";
import { ARMS, rowKey, type Arm, type JudgeResult, type ResultRow } from "./results.js";
import { scoreRow } from "./summary.js";
import type { EvalTask } from "./tasks.js";

const ARM_LABEL: Record<Arm, string> = { raw: "A raw", plan: "B plan", loop: "C loop" };

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

const STYLE = `
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6b6b70; --line:#dcdce0; --pass:#1a7f37; --fail:#c62828; --card:#ffffff; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#ececf0; --muted:#9a9aa2; --line:#33333a; --pass:#4ac26b; --fail:#ff6b6b; --card:#1e1e22; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 0 0 4px; }
.meta { color: var(--muted); margin: 0 0 12px; }
.task { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 0 0 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; margin-top: 8px; }
.cell { border-top: 1px solid var(--line); padding-top: 8px; min-width: 0; }
.thumb { display: block; max-width: 100%; height: auto; border: 1px solid var(--line); }
.checker { background-color: #fff; background-image: linear-gradient(45deg,#ccc 25%,transparent 25%),linear-gradient(-45deg,#ccc 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ccc 75%),linear-gradient(-45deg,transparent 75%,#ccc 75%); background-size: 16px 16px; background-position: 0 0,0 8px,8px -8px,-8px 0; }
.pass { color: var(--pass); font-weight: 600; } .fail { color: var(--fail); font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin-top: 6px; } td, th { border-bottom: 1px solid var(--line); padding: 3px 4px; text-align: left; vertical-align: top; }
.small { font-size: 12px; color: var(--muted); } .err { color: var(--fail); }
.disagree { background: rgba(255, 170, 0, 0.18); }
`;

function page(title: string, body: string): string {
  return `<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escapeHtml(title)}</title><style>${STYLE}</style></head><body>${body}</body></html>\n`;
}

const mark = (pass: boolean | undefined) => (pass === undefined ? "–" : pass ? '<span class="pass">pass</span>' : '<span class="fail">fail</span>');

function judgeRows(j: JudgeResult | undefined): string {
  if (!j) return '<p class="small">not graded</p>';
  if (j.error) return `<p class="err">judge error: ${escapeHtml(j.error)}</p>`;
  return `<table>${j.items.map((i) => `<tr><td>${mark(i.pass)}</td><td>${escapeHtml(i.text)}${i.evidence ? `<div class="small">${escapeHtml(i.evidence)}</div>` : ""}</td></tr>`).join("")}</table>`;
}

function rowFacts(r: ResultRow): string {
  const s = scoreRow(r);
  const checks = r.checks.map((c) => `${escapeHtml(c.id)} ${mark(c.pass)}${r.scoredChecks.includes(c.id) ? "" : " (not scored)"}`).join(" · ");
  return [
    `<div><b>${ARM_LABEL[r.arm]}</b> · seed ${r.seed} · ${s.scorable && r.status === "ok" ? `items ${s.itemsPassed}/${s.itemsTotal} · full pass ${mark(s.fullPass)}` : r.status === "ok" ? "not scored" : '<span class="err">generation failed</span>'}</div>`,
    `<div class="small">rounds ${r.rounds}${r.stopReason ? ` (${escapeHtml(r.stopReason)})` : ""} · wall ${(r.wallMs / 1000).toFixed(0)} s · ComfyUI ${(r.comfyMs / 1000).toFixed(0)} s · LLM $${r.llm.costUsd.toFixed(4)}</div>`,
    checks ? `<div class="small">code checks: ${checks}</div>` : "",
    r.error ? `<div class="err">${escapeHtml(r.error.class)}: ${escapeHtml(r.error.message)}</div>` : "",
  ].join("");
}

/** Image data URLs keyed by {@link rowKey}. */
export type Thumbs = Map<string, string>;

function img(r: ResultRow, thumbs: Thumbs, outDir: string, checker: boolean): string {
  const src = thumbs.get(rowKey(r));
  if (!src) return '<p class="small">no image</p>';
  const full = r.imagePath ? escapeHtml(path.relative(outDir, r.imagePath)) : "";
  const tag = `<img class="thumb${checker ? " checker" : ""}" src="${src}" alt="${escapeHtml(`${r.taskId} ${ARM_LABEL[r.arm]}`)}">`;
  return full ? `<a href="${full}">${tag}</a>` : tag;
}

/** All tasks, the three arms side by side, with the judge's grades under each image. */
export function renderContactSheet(tasks: EvalTask[], rows: ResultRow[], thumbs: Thumbs, outDir: string): string {
  const parts = [`<h1>Design-asset eval</h1><p class="meta">${rows.length} rows · judge ${escapeHtml(rows.find((r) => r.judge)?.judge?.model ?? "–")} · transparent assets on a checkerboard</p>`];
  for (const t of tasks) {
    const mine = rows.filter((r) => r.taskId === t.id).sort((a, b) => ARMS.indexOf(a.arm) - ARMS.indexOf(b.arm) || a.seed - b.seed);
    if (!mine.length) continue;
    parts.push(
      `<section class="task"><h2>${escapeHtml(t.id)} <span class="small">${escapeHtml(t.group)} · ${t.width}x${t.height}${t.transparent ? " · transparent" : ""}</span></h2>`,
      `<p class="meta">${escapeHtml(t.prompt)}</p><div class="grid">`,
      ...mine.map((r) => `<div class="cell">${img(r, thumbs, outDir, t.transparent)}${rowFacts(r)}${judgeRows(r.judge)}</div>`),
      "</div></section>",
    );
  }
  return page("Design-asset contact sheet", parts.join("\n"));
}

/** Reliability subset: what the judges saw, with both judges' grades side by side and disagreements highlighted. */
export function renderJudgeSheet(tasks: EvalTask[], rows: ResultRow[], views: Thumbs, outDir: string): string {
  const both = rows.filter((r) => r.judge2);
  const j1 = both.find((r) => r.judge)?.judge?.model ?? "judge 1";
  const j2 = both[0]?.judge2?.model ?? "judge 2";
  const parts = [
    `<h1>Judge spot-check</h1><p class="meta">${both.length} images graded by ${escapeHtml(j1)} and ${escapeHtml(j2)}. ` +
      "Each image is exactly what the judges were shown (cut-outs on white and near-black). Highlighted rows are disagreements.</p>",
  ];
  for (const t of tasks) {
    const mine = both.filter((r) => r.taskId === t.id).sort((a, b) => ARMS.indexOf(a.arm) - ARMS.indexOf(b.arm) || a.seed - b.seed);
    if (!mine.length) continue;
    parts.push(`<section class="task"><h2>${escapeHtml(t.id)}</h2><p class="meta">${escapeHtml(t.prompt)}</p><div class="grid">`);
    for (const r of mine) {
      const a = r.judge;
      const b = r.judge2;
      const lines = t.checklist.map((text, i) => {
        const x = a?.items[i];
        const y = b?.items[i];
        const dis = x && y && x.pass !== y.pass;
        return `<tr${dis ? ' class="disagree"' : ""}><td>${escapeHtml(text)}</td><td>${mark(x?.pass)}${x?.evidence ? `<div class="small">${escapeHtml(x.evidence)}</div>` : ""}</td><td>${mark(y?.pass)}${y?.evidence ? `<div class="small">${escapeHtml(y.evidence)}</div>` : ""}</td></tr>`;
      });
      const errs = [a?.error && `${j1}: ${a.error}`, b?.error && `${j2}: ${b.error}`].filter(Boolean).map((e) => `<p class="err">${escapeHtml(String(e))}</p>`).join("");
      const src = views.get(rowKey(r));
      parts.push(
        `<div class="cell">${src ? `<img class="thumb" src="${src}" alt="${escapeHtml(`${r.taskId} ${ARM_LABEL[r.arm]}`)}">` : '<p class="small">no image</p>'}` +
          `<div><b>${ARM_LABEL[r.arm]}</b> · seed ${r.seed}${r.imagePath ? ` · <a href="${escapeHtml(path.relative(outDir, r.imagePath))}">full image</a>` : ""}</div>${errs}` +
          `<table><tr><th>Item</th><th>${escapeHtml(j1)}</th><th>${escapeHtml(j2)}</th></tr>${lines.join("")}</table></div>`,
      );
    }
    parts.push("</div></section>");
  }
  return page("Judge spot-check", parts.join("\n"));
}

async function dataUrl(file: string | undefined, maxSide: number): Promise<string | undefined> {
  if (!file) return undefined;
  try {
    return `data:image/png;base64,${encodePng(fitWithin(decodePng(await readFile(file)), maxSide)).toString("base64")}`;
  } catch {
    return undefined;
  }
}

export async function loadThumbs(rows: ResultRow[], pick: (r: ResultRow) => string | undefined, maxSide: number): Promise<Thumbs> {
  const out: Thumbs = new Map();
  for (const r of rows) {
    const url = await dataUrl(pick(r), maxSide);
    if (url) out.set(rowKey(r), url);
  }
  return out;
}
