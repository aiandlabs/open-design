// Calibrate the matte_residue limits in src/loop.ts on real BiRefNet mattes.
//
//   pnpm --filter @open-design/comfy-agent exec tsx evals/design-assets/calibrate-matte.ts --masks <dir> --rounds <rounds.json> --out <dir>
//
// <dir>/index.json maps n -> source round image; <dir>/<n>.rgba.png is that
// image's RGB with the BiRefNet foreground as alpha (made offline with the
// same weights ComfyUI loads, by ComfyUI's own loader). For each round this
// applies the loop's matte post-processing and records the residue measures,
// next to the chroma-key result the same round got, and writes a contact
// sheet per task (cut-outs on white and near-black) for labelling.

import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";

import { contactSheet, cutoutSheet, decodePng, encodePng, estimateKeyColor, keyResidue, matteResidue, parseHexColor, decontaminateMatte } from "../../src/image.js";

const { values } = parseArgs({ options: { masks: { type: "string" }, rounds: { type: "string" }, out: { type: "string" } } });
if (!values.masks || !values.rounds || !values.out) throw new Error("--masks, --rounds and --out are required");
await mkdir(values.out, { recursive: true });

type Round = { run: string; task: string; arm: string; seed: number; raw: string; final: string; trace: string };
const rounds = JSON.parse(await readFile(values.rounds, "utf8")) as Round[];
const index = JSON.parse(await readFile(path.join(values.masks, "index.json"), "utf8")) as Record<string, string>;

const rows: Record<string, unknown>[] = [];
const sheets = new Map<string, ReturnType<typeof decodePng>[]>();
for (const [n, src] of Object.entries(index)) {
  const round = rounds.find((r) => r.raw === src);
  if (!round) continue;
  const matted = decodePng(await readFile(path.join(values.masks, `${n}.rgba.png`)));
  const raw = decodePng(await readFile(src));
  // The round's requested key colour is the backdrop the subject was drawn on.
  // Loop traces store plan.keyColor; raw-arm traces store keyColor as { requested, measured, ... }.
  const trace = JSON.parse(await readFile(round.trace, "utf8")) as { plan?: { keyColor?: string }; keyColor?: string | { requested?: string } };
  const rawKey = typeof trace.keyColor === "string" ? trace.keyColor : trace.keyColor?.requested;
  const nominal = trace.plan?.keyColor ?? rawKey ?? "#00FF00";
  const est = estimateKeyColor(raw, parseHexColor(nominal));
  const cut = decontaminateMatte(matted, est.color);
  const m = matteResidue(cut, est.color);
  let chroma: { keyTinted: number; haze: number } | undefined;
  try {
    chroma = keyResidue(decodePng(await readFile(round.final)), est.color);
  } catch {
    chroma = undefined;
  }
  rows.push({ n: Number(n), run: round.run, task: round.task, arm: round.arm, seed: round.seed, backdrop: nominal, ...m, chroma });
  const key = `${round.run}-${round.task}`;
  sheets.set(key, [...(sheets.get(key) ?? []), cutoutSheet(cut)]);
}
for (const [key, imgs] of sheets) await writeFile(path.join(values.out, `${key}.png`), encodePng(contactSheet(imgs, 320, 3, { r: 128, g: 128, b: 128 })));
await writeFile(path.join(values.out, "matte-residue.json"), JSON.stringify(rows, null, 1));
console.log(`${rows.length} rounds, ${sheets.size} sheets`);
