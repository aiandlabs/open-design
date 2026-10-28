// Read colour emoji bitmaps from an `sbix` font (Apple Color Emoji and other
// bitmap emoji fonts): each glyph in a strike is stored as a PNG. Only code
// points the cmap maps directly are found; sequences that need GSUB ligature
// lookups (ZWJ sequences, flags, keycaps) return nothing, and callers fall
// back to another emoji source. The font is only read; nothing is copied into
// the repository.

import { ImageFormatError } from "./errors.js";

export type SbixFont = {
  /** Pixels per em of the strike used (the largest in the font). */
  strikePpem: number;
  /** PNG bytes for an emoji given as code points, or undefined when the font has no direct glyph. */
  png(codepoints: number[]): Buffer | undefined;
};

const VS16 = 0xfe0f;

export function openSbixFont(bytes: Buffer, fontIndex = 0): SbixFont {
  const tag = (at: number) => bytes.toString("latin1", at, at + 4);
  let base = 0;
  if (tag(0) === "ttcf") {
    const count = bytes.readUInt32BE(8);
    if (fontIndex >= count) throw new ImageFormatError(`font collection has ${count} fonts`);
    base = bytes.readUInt32BE(12 + 4 * fontIndex);
  }
  const numTables = bytes.readUInt16BE(base + 4);
  const tables = new Map<string, { offset: number; length: number }>();
  for (let t = 0; t < numTables; t += 1) {
    const rec = base + 12 + 16 * t;
    tables.set(tag(rec), { offset: bytes.readUInt32BE(rec + 8), length: bytes.readUInt32BE(rec + 12) });
  }
  const cmap = tables.get("cmap");
  const maxp = tables.get("maxp");
  const sbix = tables.get("sbix");
  if (!cmap || !maxp || !sbix) throw new ImageFormatError("not an sbix colour font (needs cmap, maxp and sbix tables)");
  const numGlyphs = bytes.readUInt16BE(maxp.offset + 4);

  // cmap: prefer a format 12 subtable (full Unicode); emoji live outside the BMP.
  const groups: { start: number; end: number; glyph: number }[] = [];
  const nSub = bytes.readUInt16BE(cmap.offset + 2);
  for (let s = 0; s < nSub; s += 1) {
    const rec = cmap.offset + 4 + 8 * s;
    const sub = cmap.offset + bytes.readUInt32BE(rec + 4);
    if (bytes.readUInt16BE(sub) !== 12) continue;
    const n = bytes.readUInt32BE(sub + 12);
    for (let g = 0; g < n; g += 1) {
      const at = sub + 16 + 12 * g;
      groups.push({ start: bytes.readUInt32BE(at), end: bytes.readUInt32BE(at + 4), glyph: bytes.readUInt32BE(at + 8) });
    }
    break;
  }
  if (groups.length === 0) throw new ImageFormatError("font has no format 12 cmap subtable");
  groups.sort((a, b) => a.start - b.start);
  const glyphOf = (cp: number): number | undefined => {
    let lo = 0;
    let hi = groups.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const g = groups[mid]!;
      if (cp < g.start) hi = mid - 1;
      else if (cp > g.end) lo = mid + 1;
      else return g.glyph + (cp - g.start);
    }
    return undefined;
  };

  // sbix: use the largest strike.
  const nStrikes = bytes.readUInt32BE(sbix.offset + 4);
  let strike = -1;
  let ppem = 0;
  for (let s = 0; s < nStrikes; s += 1) {
    const at = sbix.offset + bytes.readUInt32BE(sbix.offset + 8 + 4 * s);
    const p = bytes.readUInt16BE(at);
    if (p > ppem) {
      ppem = p;
      strike = at;
    }
  }
  if (strike < 0) throw new ImageFormatError("sbix table has no strikes");

  const glyphPng = (glyph: number, depth = 0): Buffer | undefined => {
    if (glyph <= 0 || glyph >= numGlyphs || depth > 4) return undefined;
    const start = bytes.readUInt32BE(strike + 4 + 4 * glyph);
    const end = bytes.readUInt32BE(strike + 4 + 4 * (glyph + 1));
    if (end - start <= 8) return undefined;
    const type = tag(strike + start + 4);
    if (type === "dupe") return glyphPng(bytes.readUInt16BE(strike + start + 8), depth + 1);
    if (type !== "png ") return undefined;
    return Buffer.from(bytes.subarray(strike + start + 8, strike + end));
  };

  return {
    strikePpem: ppem,
    png(codepoints) {
      const cps = codepoints.at(-1) === VS16 ? codepoints.slice(0, -1) : codepoints;
      if (cps.length !== 1) return undefined;
      const glyph = glyphOf(cps[0]!);
      return glyph === undefined ? undefined : glyphPng(glyph);
    },
  };
}
