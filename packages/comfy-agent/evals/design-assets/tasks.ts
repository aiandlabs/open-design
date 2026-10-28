// The design-asset task set: loading, validation and selection.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export type EvalTask = {
  id: string;
  group: string;
  width: number;
  height: number;
  transparent: boolean;
  prompt: string;
  /** Fixed, human-written checklist. The judge grades exactly these items. */
  checklist: string[];
  /**
   * Icon set this task belongs to. Members are generated in file order; each
   * gets the earlier members (same arm and seed) as style references, and the
   * finished set is judged for consistency as a whole.
   */
  set?: string;
};

export const TASKS_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), "tasks.json");

/**
 * Tasks graded by the second judge as well. Fixed so that agreement numbers
 * from different runs are comparable: it covers every group and includes
 * three transparent tasks.
 */
export const DEFAULT_RELIABILITY_SUBSET = [
  "hero-tokyo-dawn",
  "hero-fintech-abstract",
  "tex-washi",
  "bg-night-sky",
  "icon-lantern",
  "illo-404",
  "illo-map-pin",
  "prod-sneaker",
] as const;

export function parseTasks(raw: unknown): EvalTask[] {
  const doc = raw as { tasks?: unknown };
  if (!doc || !Array.isArray(doc.tasks)) throw new Error("tasks file must have a \"tasks\" array");
  const seen = new Set<string>();
  return doc.tasks.map((t: unknown, i: number) => {
    const o = (t ?? {}) as Record<string, unknown>;
    const where = `tasks[${i}]`;
    if (typeof o.id !== "string" || !o.id) throw new Error(`${where}.id must be a non-empty string`);
    if (seen.has(o.id)) throw new Error(`duplicate task id ${o.id}`);
    seen.add(o.id);
    if (typeof o.group !== "string" || !o.group) throw new Error(`${o.id}: group must be a string`);
    for (const k of ["width", "height"] as const) {
      if (!Number.isInteger(o[k]) || (o[k] as number) < 16) throw new Error(`${o.id}: ${k} must be an integer >= 16`);
    }
    if (typeof o.prompt !== "string" || !o.prompt.trim()) throw new Error(`${o.id}: prompt must be a string`);
    if (!Array.isArray(o.checklist) || o.checklist.length === 0 || !o.checklist.every((c) => typeof c === "string" && c.trim())) {
      throw new Error(`${o.id}: checklist must be a non-empty array of strings`);
    }
    if (o.set !== undefined && (typeof o.set !== "string" || !o.set.trim())) throw new Error(`${o.id}: set must be a non-empty string`);
    return {
      id: o.id,
      group: o.group,
      width: o.width as number,
      height: o.height as number,
      transparent: o.transparent === true,
      prompt: o.prompt.trim(),
      checklist: (o.checklist as string[]).map((c) => c.trim()),
      ...(typeof o.set === "string" ? { set: o.set.trim() } : {}),
    };
  });
}

export async function loadTasks(file = TASKS_PATH): Promise<EvalTask[]> {
  return parseTasks(JSON.parse(await readFile(file, "utf8")));
}

/** Filter by explicit ids and/or groups, keeping file order. Unknown ids or groups are an error. */
export function selectTasks(all: EvalTask[], filter: { ids?: string[]; groups?: string[] }): EvalTask[] {
  const ids = filter.ids?.filter(Boolean) ?? [];
  const groups = filter.groups?.filter(Boolean) ?? [];
  const unknownIds = ids.filter((id) => !all.some((t) => t.id === id));
  if (unknownIds.length) throw new Error(`unknown task id(s): ${unknownIds.join(", ")}`);
  const unknownGroups = groups.filter((g) => !all.some((t) => t.group === g));
  if (unknownGroups.length) throw new Error(`unknown group(s): ${unknownGroups.join(", ")}`);
  return all.filter((t) => (ids.length === 0 || ids.includes(t.id)) && (groups.length === 0 || groups.includes(t.group)));
}

/** Stable checklist item ids the judge answers with. */
export function checklistIds(task: EvalTask): string[] {
  return task.checklist.map((_, i) => `c${i + 1}`);
}
