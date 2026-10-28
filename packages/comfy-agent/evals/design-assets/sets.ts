// Icon sets: style references for a set member, and the whole-set
// consistency judgement. Set members are tasks with the same `set`, generated
// in task-file order; member k gets members 1..k-1 of the same arm and seed
// as style references (the loop-noset arm gets none). When every member of a
// set has an image, the judge grades a contact sheet of the set for
// consistency. One row per set × arm × seed in set-results.jsonl.

import { appendFile, readFile } from "node:fs/promises";

import { contactSheet, encodePng, type RgbaImage } from "../../src/image.js";
import type { JsonSchema } from "../../src/llm.js";
import type { SetReference } from "../../src/loop.js";
import type { JudgeOptions } from "./judge.js";
import type { Arm, ResultRow } from "./results.js";
import type { EvalTask } from "./tasks.js";

/** Most earlier members passed to one generation (the loop's own default). */
export const MAX_SET_REFERENCES = 3;

export type SetJudgement = {
  set: string;
  arm: Arm;
  seed: number;
  members: string[];
  model: string;
  /** All members read as one family. */
  consistent?: boolean;
  /** 1 (unrelated styles) to 5 (indistinguishable style). */
  score?: number;
  /** Members the judge named as off-style. */
  outliers?: string[];
  evidence?: string;
  error?: string;
  costUsd: number;
  finishedAt: string;
};

/** Members of a set in task-file order. */
export function setMembers(tasks: EvalTask[], set: string): EvalTask[] {
  return tasks.filter((t) => t.set === set);
}

/**
 * Earlier members of the task's set that already have an image from the same
 * arm and seed, most recent first, as the loop's set references.
 */
export function setReferencesFor(
  task: EvalTask,
  arm: Arm,
  seed: number,
  tasks: EvalTask[],
  rows: ResultRow[],
  readImage: (path: string) => Buffer | undefined,
): SetReference[] {
  if (!task.set || arm === "loop-noset") return [];
  const members = setMembers(tasks, task.set);
  const earlier = members.slice(0, members.findIndex((m) => m.id === task.id));
  const out: SetReference[] = [];
  for (const m of earlier.reverse()) {
    const row = rows.find((r) => r.taskId === m.id && r.arm === arm && r.seed === seed && r.status === "ok" && r.imagePath);
    const png = row?.imagePath ? readImage(row.imagePath) : undefined;
    if (png) out.push({ id: m.id, png, label: m.prompt });
    if (out.length >= MAX_SET_REFERENCES) break;
  }
  return out;
}

export const SET_JUDGE_TOOL = "submit_set_judgement";

export const SET_JUDGE_SYSTEM = `You are a strict visual QA reviewer for icon sets. You receive one image that shows several icons side by side, separated by thin magenta lines that are not part of any icon, each on a plain background.
Decide whether the icons read as one family: the same rendering technique, palette, outline weight, corner rounding, lighting and level of detail. Different subjects are expected; judge only the style.
Score 5 when the styles are indistinguishable, 4 when one detail differs, 3 when a viewer would notice a mismatch, 2 when two or more icons clearly differ, 1 when there is no common style. "consistent" is true only for 4 or 5. Name every icon that is off-style by its position (1 = first, left to right, top to bottom).`;

export const SET_JUDGE_SCHEMA: JsonSchema = {
  type: "object",
  properties: {
    consistent: { type: "boolean" },
    score: { type: "integer", minimum: 1, maximum: 5 },
    outliers: { type: "array", items: { type: "integer" } },
    evidence: { type: "string" },
  },
  required: ["consistent", "score", "evidence"],
};

/** Grade a finished set for consistency. Never throws; a failed judge returns `error`. */
export async function judgeSet(
  set: string,
  arm: Arm,
  seed: number,
  members: { id: string; image: RgbaImage }[],
  options: JudgeOptions,
  now = () => new Date(),
): Promise<SetJudgement> {
  const base = { set, arm, seed, members: members.map((m) => m.id), model: options.model, costUsd: 0 };
  try {
    const sheet = encodePng(contactSheet(members.map((m) => m.image), 320, 4, { r: 245, g: 245, b: 245 }));
    const res = await options.llm.structured<Record<string, unknown>>(
      {
        model: options.model,
        system: SET_JUDGE_SYSTEM,
        content: [
          { type: "image", mime: "image/png", base64: sheet.toString("base64") },
          { type: "text", text: `${members.length} icons from one set. Grade their style consistency and call ${SET_JUDGE_TOOL}.` },
        ],
        schema: SET_JUDGE_SCHEMA,
        toolName: SET_JUDGE_TOOL,
        maxTokens: 2048,
        ...(options.reasoningEffort ? { reasoningEffort: options.reasoningEffort } : {}),
      },
      options.shape,
    );
    const d = res.data;
    const score = Number(d.score);
    const outliers = Array.isArray(d.outliers)
      ? d.outliers.map(Number).filter((n) => Number.isInteger(n) && n >= 1 && n <= members.length).map((n) => members[n - 1]!.id)
      : [];
    return {
      ...base,
      ...(Number.isInteger(score) && score >= 1 && score <= 5 ? { score, consistent: score >= 4 } : typeof d.consistent === "boolean" ? { consistent: d.consistent } : {}),
      outliers,
      ...(typeof d.evidence === "string" ? { evidence: d.evidence } : {}),
      costUsd: res.reportedCostUsd ?? res.estimatedCostUsd ?? 0,
      finishedAt: now().toISOString(),
    };
  } catch (err) {
    return { ...base, error: err instanceof Error ? err.message.slice(0, 500) : String(err), finishedAt: now().toISOString() };
  }
}

export async function readSetResults(file: string): Promise<SetJudgement[]> {
  let text: string;
  try {
    text = await readFile(file, "utf8");
  } catch {
    return [];
  }
  const out = new Map<string, SetJudgement>();
  for (const line of text.split("\n")) {
    try {
      const j = JSON.parse(line) as SetJudgement;
      if (j && typeof j.set === "string") out.set(`${j.set}|${j.arm}|${j.seed}`, j);
    } catch {
      // A line cut short by a crash.
    }
  }
  return [...out.values()];
}

export async function appendSetResult(file: string, j: SetJudgement): Promise<void> {
  await appendFile(file, `${JSON.stringify(j)}\n`);
}

/** Per arm: sets judged, share judged consistent, and mean score. */
export function setConsistencyByArm(judgements: SetJudgement[]): { arm: Arm; sets: number; consistentRate: number | null; meanScore: number | null }[] {
  const arms = [...new Set(judgements.map((j) => j.arm))];
  return arms.map((arm) => {
    const ok = judgements.filter((j) => j.arm === arm && !j.error && typeof j.score === "number");
    return {
      arm,
      sets: ok.length,
      consistentRate: ok.length ? ok.filter((j) => j.consistent).length / ok.length : null,
      meanScore: ok.length ? ok.reduce((s, j) => s + j.score!, 0) / ok.length : null,
    };
  });
}
