// Scoring and aggregation: results.jsonl rows → summary.json and summary.md.

import { ARMS, type Arm, type JudgeResult, type ResultRow } from "./results.js";
import { cohensKappa, comparePaired, mean, median, type Agreement, type PairedComparison } from "./stats.js";

export type RowScore = {
  /** False when the row cannot be scored (the judge failed); such rows are excluded and counted. */
  scorable: boolean;
  itemPassRate: number;
  itemsPassed: number;
  itemsTotal: number;
  codePass: boolean;
  fullPass: boolean;
};

function judgeUsable(j: JudgeResult | undefined): j is JudgeResult {
  return Boolean(j && !j.error && j.items.length > 0);
}

/**
 * A generation failure scores zero: the arm produced nothing usable. A judge
 * failure is an eval failure, not the arm's, so the row is left out.
 */
export function scoreRow(row: ResultRow, judge: "judge" | "judge2" = "judge"): RowScore {
  if (row.status !== "ok") return { scorable: true, itemPassRate: 0, itemsPassed: 0, itemsTotal: 0, codePass: false, fullPass: false };
  const j = row[judge];
  if (!judgeUsable(j)) return { scorable: false, itemPassRate: 0, itemsPassed: 0, itemsTotal: 0, codePass: false, fullPass: false };
  const itemsPassed = j.items.filter((i) => i.pass).length;
  const codePass = row.scoredChecks.every((id) => row.checks.find((c) => c.id === id)?.pass === true);
  return {
    scorable: true,
    itemPassRate: itemsPassed / j.items.length,
    itemsPassed,
    itemsTotal: j.items.length,
    codePass,
    fullPass: codePass && itemsPassed === j.items.length,
  };
}

type Dist = { mean: number | null; median: number | null };
const dist = (xs: number[]): Dist => ({ mean: mean(xs), median: median(xs) });

export type ArmStats = {
  arm: Arm;
  rows: number;
  scored: number;
  generationErrors: number;
  judgeErrors: number;
  fullPassRate: number | null;
  meanItemPassRate: number | null;
  codeCheckPassRate: number | null;
  rounds: Dist;
  wallSeconds: Dist;
  /** Wall time minus time queued behind other ComfyUI clients (rows with history timing only). */
  wallExclQueueSeconds: Dist;
  comfySeconds: Dist;
  /** ComfyUI execution time from server history (rows with history timing only). */
  comfyExecSeconds: Dist;
  /** Total time queued behind other ComfyUI clients. */
  queueWaitSecondsTotal: number;
  stopReasons: Record<string, number>;
  costUsd: Dist;
  totalCostUsd: number;
};

export function armStats(arm: Arm, rows: ResultRow[]): ArmStats {
  const mine = rows.filter((r) => r.arm === arm);
  const scored = mine.map((r) => ({ r, s: scoreRow(r) })).filter((x) => x.s.scorable);
  const ok = mine.filter((r) => r.status === "ok");
  const rate = (xs: boolean[]) => (xs.length ? xs.filter(Boolean).length / xs.length : null);
  return {
    arm,
    rows: mine.length,
    scored: scored.length,
    generationErrors: mine.filter((r) => r.status === "error").length,
    judgeErrors: mine.filter((r) => r.status === "ok" && !judgeUsable(r.judge)).length,
    fullPassRate: rate(scored.map((x) => x.s.fullPass)),
    meanItemPassRate: mean(scored.map((x) => x.s.itemPassRate)),
    codeCheckPassRate: rate(scored.map((x) => x.s.codePass)),
    rounds: dist(ok.map((r) => r.rounds)),
    wallSeconds: dist(mine.map((r) => r.wallMs / 1000)),
    wallExclQueueSeconds: dist(ok.filter((r) => r.comfyQueueMs !== undefined).map((r) => (r.wallMs - r.comfyQueueMs!) / 1000)),
    comfySeconds: dist(ok.map((r) => r.comfyMs / 1000)),
    comfyExecSeconds: dist(ok.filter((r) => r.comfyExecMs !== undefined).map((r) => r.comfyExecMs! / 1000)),
    queueWaitSecondsTotal: ok.reduce((s, r) => s + (r.comfyQueueMs ?? 0), 0) / 1000,
    stopReasons: mine.reduce<Record<string, number>>((acc, r) => {
      const k = r.status === "error" ? "error" : (r.stopReason ?? "unknown");
      acc[k] = (acc[k] ?? 0) + 1;
      return acc;
    }, {}),
    costUsd: dist(mine.map((r) => r.llm.costUsd)),
    totalCostUsd: mine.reduce((s, r) => s + r.llm.costUsd, 0),
  };
}

/** Per-task value for an arm, averaged over seeds; undefined when the arm has no scorable row for the task. */
function perTask(rows: ResultRow[], arm: Arm, pick: (s: RowScore) => number): Map<string, number> {
  const acc = new Map<string, number[]>();
  for (const r of rows) {
    if (r.arm !== arm) continue;
    const s = scoreRow(r);
    if (!s.scorable) continue;
    acc.set(r.taskId, [...(acc.get(r.taskId) ?? []), pick(s)]);
  }
  return new Map([...acc].map(([k, v]) => [k, mean(v)!]));
}

export type PairedReport = { a: Arm; b: Arm; itemPassRate: PairedComparison; fullPass: PairedComparison };

export function pairedReport(rows: ResultRow[], a: Arm, b: Arm, seed = 12345): PairedReport {
  const build = (pick: (s: RowScore) => number) => {
    const va = perTask(rows, a, pick);
    const vb = perTask(rows, b, pick);
    const pairs = [...va.keys()].filter((k) => vb.has(k)).sort().map((k) => ({ a: va.get(k)!, b: vb.get(k)! }));
    return comparePaired(pairs, { seed });
  };
  return { a, b, itemPassRate: build((s) => s.itemPassRate), fullPass: build((s) => (s.fullPass ? 1 : 0)) };
}

export type JudgeAgreementReport = {
  judge: string | null;
  judge2: string | null;
  images: number;
  items: Agreement;
  fullPass: Agreement;
  byArm: { arm: Arm; items: Agreement; meanItemPassRateJudge: number | null; meanItemPassRateJudge2: number | null }[];
  byTransparency: { transparent: boolean; items: Agreement }[];
  /** Per checklist item (task × item): how many graded images the two judges agreed on. */
  perItem: { taskId: string; itemId: string; text: string; n: number; agree: number }[];
  judgeCostUsd: number;
  judge2CostUsd: number;
};

export function judgeAgreement(rows: ResultRow[]): JudgeAgreementReport {
  const both = rows.filter((r) => r.status === "ok" && judgeUsable(r.judge) && judgeUsable(r.judge2));
  const pairsOf = (rs: ResultRow[]) => {
    const a: boolean[] = [];
    const b: boolean[] = [];
    for (const r of rs) {
      for (const it of r.judge!.items) {
        const other = r.judge2!.items.find((x) => x.id === it.id);
        if (!other) continue;
        a.push(it.pass);
        b.push(other.pass);
      }
    }
    return cohensKappa(a, b);
  };
  const perItemMap = new Map<string, { taskId: string; itemId: string; text: string; n: number; agree: number }>();
  for (const r of both) {
    for (const it of r.judge!.items) {
      const other = r.judge2!.items.find((x) => x.id === it.id);
      if (!other) continue;
      const k = `${r.taskId}|${it.id}`;
      const cur = perItemMap.get(k) ?? { taskId: r.taskId, itemId: it.id, text: it.text, n: 0, agree: 0 };
      cur.n += 1;
      if (it.pass === other.pass) cur.agree += 1;
      perItemMap.set(k, cur);
    }
  }
  const ipr = (j: JudgeResult) => j.items.filter((i) => i.pass).length / j.items.length;
  return {
    judge: rows.find((r) => r.judge)?.judge?.model ?? null,
    judge2: rows.find((r) => r.judge2)?.judge2?.model ?? null,
    images: both.length,
    items: pairsOf(both),
    fullPass: cohensKappa(
      both.map((r) => scoreRow(r, "judge").fullPass),
      both.map((r) => scoreRow(r, "judge2").fullPass),
    ),
    byArm: ARMS.map((arm) => {
      const rs = both.filter((r) => r.arm === arm);
      return {
        arm,
        items: pairsOf(rs),
        meanItemPassRateJudge: mean(rs.map((r) => ipr(r.judge!))),
        meanItemPassRateJudge2: mean(rs.map((r) => ipr(r.judge2!))),
      };
    }),
    byTransparency: [false, true].map((t) => ({ transparent: t, items: pairsOf(both.filter((r) => r.transparent === t)) })),
    perItem: [...perItemMap.values()],
    judgeCostUsd: rows.reduce((s, r) => s + (r.judge?.costUsd ?? 0), 0),
    judge2CostUsd: rows.reduce((s, r) => s + (r.judge2?.costUsd ?? 0), 0),
  };
}

export type Summary = {
  generatedAt: string;
  rows: number;
  tasks: number;
  arms: Arm[];
  seeds: number[];
  perArm: ArmStats[];
  perGroup: { group: string; perArm: ArmStats[] }[];
  paired: PairedReport[];
  judgeAgreement: JudgeAgreementReport;
  errors: { taskId: string; arm: Arm; seed: number; stage: string; class: string; message: string }[];
};

export function summarize(rows: ResultRow[], now = new Date()): Summary {
  const arms = ARMS.filter((a) => rows.some((r) => r.arm === a));
  const groups = [...new Set(rows.map((r) => r.group))];
  const paired: PairedReport[] = [];
  if (arms.includes("loop") && arms.includes("raw")) paired.push(pairedReport(rows, "raw", "loop"));
  if (arms.includes("loop") && arms.includes("plan")) paired.push(pairedReport(rows, "plan", "loop"));
  if (arms.includes("plan") && arms.includes("raw")) paired.push(pairedReport(rows, "raw", "plan"));
  const errors: Summary["errors"] = [];
  for (const r of rows) {
    if (r.error) errors.push({ taskId: r.taskId, arm: r.arm, seed: r.seed, stage: r.error.stage, class: r.error.class, message: r.error.message });
    for (const j of ["judge", "judge2"] as const) {
      const e = r[j]?.error;
      if (e) errors.push({ taskId: r.taskId, arm: r.arm, seed: r.seed, stage: j, class: "JudgeError", message: e });
    }
  }
  return {
    generatedAt: now.toISOString(),
    rows: rows.length,
    tasks: new Set(rows.map((r) => r.taskId)).size,
    arms,
    seeds: [...new Set(rows.map((r) => r.seed))].sort((a, b) => a - b),
    perArm: arms.map((a) => armStats(a, rows)),
    perGroup: groups.map((g) => ({ group: g, perArm: arms.map((a) => armStats(a, rows.filter((r) => r.group === g))) })),
    paired,
    judgeAgreement: judgeAgreement(rows),
    errors,
  };
}

// ---------------------------------------------------------------------------
// Markdown

const ARM_LABEL: Record<Arm, string> = { raw: "A raw", plan: "B plan", loop: "C loop" };
const pct = (v: number | null) => (v === null ? "–" : `${(v * 100).toFixed(1)}%`);
const num = (v: number | null, d = 1) => (v === null ? "–" : v.toFixed(d));
const usd = (v: number | null) => (v === null ? "–" : `$${v.toFixed(4)}`);
const signed = (v: number | null) => (v === null ? "–" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)} pp`);

function armTable(stats: ArmStats[]): string[] {
  return [
    "| Arm | Rows | Scored | Gen errors | Judge errors | Full pass | Mean item pass | Code checks pass | Rounds mean / median | Wall s mean / median | Wall s excl. queue mean / median | ComfyUI exec s mean / median | LLM cost mean / median | LLM cost total | Stop reasons |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ...stats.map(
      (s) =>
        `| ${ARM_LABEL[s.arm]} | ${s.rows} | ${s.scored} | ${s.generationErrors} | ${s.judgeErrors} | ${pct(s.fullPassRate)} | ${pct(s.meanItemPassRate)} | ${pct(s.codeCheckPassRate)} | ` +
        `${num(s.rounds.mean, 2)} / ${num(s.rounds.median, 1)} | ${num(s.wallSeconds.mean)} / ${num(s.wallSeconds.median)} | ` +
        `${num(s.wallExclQueueSeconds.mean)} / ${num(s.wallExclQueueSeconds.median)} | ${num(s.comfyExecSeconds.mean)} / ${num(s.comfyExecSeconds.median)} | ` +
        `${usd(s.costUsd.mean)} / ${usd(s.costUsd.median)} | ${usd(s.totalCostUsd)} | ${Object.entries(s.stopReasons).map(([k, v]) => `${k} ${v}`).join(", ")} |`,
    ),
  ];
}

const kappaCell = (a: Agreement) => `${a.n} | ${pct(a.po)} | ${a.kappa === null ? "undefined" : a.kappa.toFixed(3)}`;

export function renderSummaryMarkdown(s: Summary): string {
  const out: string[] = [
    "# Design-asset eval: summary",
    "",
    `Generated ${s.generatedAt}. ${s.rows} rows, ${s.tasks} tasks, arms ${s.arms.map((a) => ARM_LABEL[a]).join(", ")}, seeds ${s.seeds.join(", ")}.`,
    `Judge: ${s.judgeAgreement.judge ?? "–"}. Second judge (reliability subset): ${s.judgeAgreement.judge2 ?? "–"}.`,
    "",
    "Definitions: *item pass* is the share of the task's 5 fixed checklist items the judge passed. *Full pass* is all 5 items plus the code checks " +
      "(exact size; alpha and key residue when the task needs transparency). A generation error scores 0 and no full pass; a row the judge could not grade is left out. " +
      "Cost is the arm's own planner and critic LLM cost; judge cost is listed separately. " +
      "*Wall s* is measured by the runner and includes time queued behind other clients of the same ComfyUI server; *excl. queue* subtracts that wait " +
      "(ComfyUI time seen by the client minus execution time from the server's history timestamps). *ComfyUI exec s* is execution time only.",
    "",
    "## Per arm",
    "",
    ...armTable(s.perArm),
    "",
    "## Per group",
    "",
  ];
  for (const g of s.perGroup) {
    out.push(`### ${g.group}`, "", ...armTable(g.perArm), "");
  }
  out.push(
    `Time queued behind other ComfyUI clients: ${s.perArm.map((a) => `${ARM_LABEL[a.arm]} ${num(a.queueWaitSecondsTotal, 0)} s`).join(", ")}.`,
    "",
    "## Paired comparisons",
    "",
    "Per task (mean over seeds), second arm minus first. Sign test: exact two-sided binomial on improved vs regressed, ties dropped. CI: 95% percentile bootstrap over tasks (10,000 resamples) of the mean difference.",
    "",
    "| Comparison | Measure | Tasks | Improved | Tied | Regressed | Mean diff | 95% CI | Sign test p |",
    "|---|---|---:|---:|---:|---:|---:|---|---:|",
  );
  for (const p of s.paired) {
    for (const [label, c] of [["item pass rate", p.itemPassRate], ["full pass", p.fullPass]] as const) {
      out.push(
        `| ${ARM_LABEL[p.b]} vs ${ARM_LABEL[p.a]} | ${label} | ${c.n} | ${c.improved} | ${c.tied} | ${c.regressed} | ${signed(c.meanDiff)} | ` +
          `${c.ci95 ? `${signed(c.ci95.low)} … ${signed(c.ci95.high)}` : "–"} | ${c.signTestP.toFixed(3)} |`,
      );
    }
  }
  const ja = s.judgeAgreement;
  out.push(
    "",
    "## Judge agreement",
    "",
    `${ja.images} images graded by both ${ja.judge ?? "–"} and ${ja.judge2 ?? "–"}. Judge cost ${usd(ja.judgeCostUsd)}, second judge ${usd(ja.judge2CostUsd)}.`,
    "",
    "| Scope | Grades | Agreement | Cohen's kappa | Pass rate judge 1 | Pass rate judge 2 |",
    "|---|---:|---:|---:|---:|---:|",
    `| All checklist items | ${kappaCell(ja.items)} | ${pct(ja.items.passRateA)} | ${pct(ja.items.passRateB)} |`,
    `| Full pass per image | ${kappaCell(ja.fullPass)} | ${pct(ja.fullPass.passRateA)} | ${pct(ja.fullPass.passRateB)} |`,
    ...ja.byArm.map((b) => `| Items, ${ARM_LABEL[b.arm]} | ${kappaCell(b.items)} | ${pct(b.items.passRateA)} | ${pct(b.items.passRateB)} |`),
    ...ja.byTransparency.map((b) => `| Items, ${b.transparent ? "transparent" : "opaque"} tasks | ${kappaCell(b.items)} | ${pct(b.items.passRateA)} | ${pct(b.items.passRateB)} |`),
    "",
  );
  const disagreed = ja.perItem.filter((p) => p.agree < p.n);
  out.push(
    `Per checklist item: the judges agreed on every image for ${ja.perItem.length - disagreed.length} of ${ja.perItem.length} items. Items with at least one disagreement:`,
    "",
  );
  if (disagreed.length) {
    out.push("| Task | Item | Agreed / images |", "|---|---|---:|", ...disagreed.map((p) => `| ${p.taskId} | ${p.itemId}: ${p.text.replace(/\|/g, "\\|")} | ${p.agree} / ${p.n} |`));
  } else {
    out.push("None.");
  }
  out.push("", "## Errors", "");
  if (s.errors.length) {
    out.push("| Task | Arm | Seed | Stage | Class | Message |", "|---|---|---:|---|---|---|");
    for (const e of s.errors) out.push(`| ${e.taskId} | ${ARM_LABEL[e.arm]} | ${e.seed} | ${e.stage} | ${e.class} | ${e.message.replace(/\s+/g, " ").replace(/\|/g, "\\|").slice(0, 200)} |`);
  } else {
    out.push("None.");
  }
  out.push("");
  return out.join("\n");
}
