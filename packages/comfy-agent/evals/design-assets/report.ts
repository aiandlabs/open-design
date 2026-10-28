// Writes summary.json, summary.md, contact-sheet.html and judge-subset.html from result rows.

import { writeFile } from "node:fs/promises";
import path from "node:path";

import { loadThumbs, renderContactSheet, renderJudgeSheet } from "./html.js";
import { ARMS, type ResultRow } from "./results.js";
import { renderSummaryMarkdown, summarize } from "./summary.js";
import type { EvalTask } from "./tasks.js";
import { checkpointOf, llmModelsOf } from "./timing.js";

export async function writeReports(outDir: string, tasks: EvalTask[], rows: ResultRow[]): Promise<string[]> {
  // Rows written before rows carried their models get them from their trace,
  // so the summary always records the image model, planner and critic.
  const filled = await Promise.all(
    rows.map(async (r) => {
      if (r.status !== "ok") return r;
      const checkpoint = r.checkpoint ?? (await checkpointOf(r));
      const models = r.models ?? (await llmModelsOf(r));
      return { ...r, ...(checkpoint ? { checkpoint } : {}), ...(models ? { models } : {}) };
    }),
  );
  const ordered = filled.sort((a, b) => a.taskId.localeCompare(b.taskId) || ARMS.indexOf(a.arm) - ARMS.indexOf(b.arm) || a.seed - b.seed);
  const summary = summarize(ordered);
  await writeFile(path.join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  await writeFile(path.join(outDir, "summary.md"), renderSummaryMarkdown(summary));
  const thumbs = await loadThumbs(ordered, (r) => r.imagePath, 360);
  await writeFile(path.join(outDir, "contact-sheet.html"), renderContactSheet(tasks, ordered, thumbs, outDir));
  const subsetRows = ordered.filter((r) => r.judge2);
  const views = await loadThumbs(subsetRows, (r) => r.judgeViewPath, 720);
  await writeFile(path.join(outDir, "judge-subset.html"), renderJudgeSheet(tasks, subsetRows, views, outDir));
  return ["summary.md", "summary.json", "contact-sheet.html", "judge-subset.html"];
}

