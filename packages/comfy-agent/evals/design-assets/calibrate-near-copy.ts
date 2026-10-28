// Calibrate the near-copy limits in src/image.ts (NEAR_COPY_MIN_IOU,
// NEAR_COPY_MIN_STRUCTURE) on real emoji.
//
//   pnpm --filter @open-design/comfy-agent exec tsx evals/design-assets/calibrate-near-copy.ts --cache <dir> [--out <file.json>]
//
// Redraws: the same emoji drawn by different artists (Apple Color Emoji when
// the system font exists, Fluent Emoji 3D, Noto Emoji). A reference used for
// shape and concept should produce output at least this different, so none of
// these pairs may count as a near-copy.
// Copies: one emoji scaled, moved, recoloured and softened. All of these must
// count as a near-copy.
// Fetches (and caches) the pinned reference sources; does not use ComfyUI.

import { writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { compositeOver, createImage, decodePng, referenceSimilarity, resize, type RgbaImage } from "../../src/image.js";
import { ReferenceLibrary, type EmojiRenderer } from "../../src/references.js";

const SUBJECTS = [
  "2615", "1F3EE", "1F680", "1F343", "1F6E1", "1F431", "1F34E", "1F697", "2708", "1F3E0", "1F4F7", "1F381", "1F4A1", "1F511", "1F512",
  "1F514", "1F4DA", "270F", "1F3B8", "1F3A7", "1F45F", "231A", "1F4BB", "1F4F1", "1F37A", "1F355", "1F354", "1F366", "1F370", "1F36A",
  "1F335", "1F33B", "1F332", "1F308", "2602", "26A1", "1F525", "1F4A7", "2744", "1F319", "1F31E", "2B50", "1F3C6", "1F3AF", "1F3A8",
  "1F4E6", "1F6B2", "1F68C", "1F6A2", "1F3D4", "1F5FC", "26F0", "1F3D6", "1F40B", "1F98A", "1F43C", "1F427", "1F41D", "1F98B", "1F422",
];

const { values } = parseArgs({ options: { cache: { type: "string" }, out: { type: "string" } } });
if (!values.cache) throw new Error("--cache <dir> is required");

const sources: EmojiRenderer[] = ["apple", "fluent", "noto"];
const libs = Object.fromEntries(sources.map((s) => [s, new ReferenceLibrary({ cacheDir: values.cache!, emojiRenderers: [s] })])) as Record<EmojiRenderer, ReferenceLibrary>;

function transform(img: RgbaImage, kind: number): RgbaImage {
  const size = img.width;
  const out = createImage(size, size, { r: 0, g: 0, b: 0, a: 0 });
  const scale = [0.7, 0.85, 0.6, 0.9][kind % 4]!;
  const s = resize(img, Math.round(size * scale), Math.round(size * scale));
  const ox = Math.floor((size - s.width) / 2) + [0, 20, -15, 5][kind % 4]!;
  const oy = Math.floor((size - s.height) / 2) + [0, -10, 25, 0][kind % 4]!;
  for (let y = 0; y < s.height; y += 1) {
    for (let x = 0; x < s.width; x += 1) {
      const tx = x + ox;
      const ty = y + oy;
      if (tx < 0 || ty < 0 || tx >= size || ty >= size) continue;
      const i = (y * s.width + x) * 4;
      const o = (ty * size + tx) * 4;
      // Recolour: rotate channels slightly and lift brightness.
      const [r, g, b] = [s.data[i]!, s.data[i + 1]!, s.data[i + 2]!];
      out.data.set(kind % 2 ? [Math.min(255, g + 20), Math.min(255, b + 20), Math.min(255, r + 20), s.data[i + 3]!] : [r, g, b, s.data[i + 3]!], o);
    }
  }
  // Soften with a down/up resample, as a regenerated image would be.
  const soft = resize(resize(out, Math.round(size / 2), Math.round(size / 2)), size, size);
  return kind === 3 ? compositeOver(soft, { r: 255, g: 255, b: 255 }) : soft;
}

const redraws: { emoji: string; pair: string; iou: number; structure: number; nearCopy: boolean }[] = [];
const copies: { emoji: string; source: string; kind: number; iou: number; structure: number; nearCopy: boolean }[] = [];
for (const hex of SUBJECTS) {
  const imgs: Partial<Record<EmojiRenderer, RgbaImage>> = {};
  for (const s of sources) {
    try {
      imgs[s] = decodePng((await libs[s].load(`emoji:${hex}`, 256)).png);
    } catch {
      // This source lacks the emoji.
    }
  }
  const have = sources.filter((s) => imgs[s]);
  for (let i = 0; i < have.length; i += 1) {
    for (let j = i + 1; j < have.length; j += 1) {
      const sim = referenceSimilarity(imgs[have[i]!]!, imgs[have[j]!]!);
      redraws.push({ emoji: hex, pair: `${have[i]}-${have[j]}`, iou: sim.silhouetteIoU, structure: sim.structure, nearCopy: sim.nearCopy });
    }
  }
  for (const s of have) {
    for (let k = 0; k < 4; k += 1) {
      const sim = referenceSimilarity(transform(imgs[s]!, k), imgs[s]!);
      copies.push({ emoji: hex, source: s, kind: k, iou: sim.silhouetteIoU, structure: sim.structure, nearCopy: sim.nearCopy });
    }
  }
}

const q = (xs: number[], p: number) => {
  const s = xs.slice().sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))]!;
};
const summary = {
  redraws: {
    pairs: redraws.length,
    flagged: redraws.filter((r) => r.nearCopy).length,
    iou: { p50: q(redraws.map((r) => r.iou), 0.5), p95: q(redraws.map((r) => r.iou), 0.95), max: q(redraws.map((r) => r.iou), 1) },
    structure: { p50: q(redraws.map((r) => r.structure), 0.5), p95: q(redraws.map((r) => r.structure), 0.95), max: q(redraws.map((r) => r.structure), 1) },
  },
  copies: {
    pairs: copies.length,
    flagged: copies.filter((r) => r.nearCopy).length,
    iou: { min: q(copies.map((r) => r.iou), 0), p05: q(copies.map((r) => r.iou), 0.05), p50: q(copies.map((r) => r.iou), 0.5) },
    structure: { min: q(copies.map((r) => r.structure), 0), p05: q(copies.map((r) => r.structure), 0.05), p50: q(copies.map((r) => r.structure), 0.5) },
  },
};
console.log(JSON.stringify(summary, null, 2));
const flaggedRedraws = redraws.filter((r) => r.nearCopy);
if (flaggedRedraws.length) console.log("redraws flagged:", flaggedRedraws);
const missedCopies = copies.filter((r) => !r.nearCopy);
if (missedCopies.length) console.log("copies missed:", missedCopies);
if (values.out) await writeFile(values.out, JSON.stringify({ summary, redraws, copies }, null, 2));
