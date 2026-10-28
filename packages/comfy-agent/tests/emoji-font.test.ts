import { describe, expect, it } from "vitest";

import { openSbixFont } from "../src/emoji-font.js";
import { solidPng } from "./fixtures/fake-comfy.js";

/** Minimal sfnt with cmap (format 12), maxp and sbix tables, optionally wrapped in a TTC. */
function buildSbixFont(glyphs: { cp: number; png?: Buffer; dupeOf?: number }[], ttc = false): Buffer {
  const numGlyphs = glyphs.length + 1; // glyph 0 is .notdef
  const u16 = (v: number) => {
    const b = Buffer.alloc(2);
    b.writeUInt16BE(v);
    return b;
  };
  const u32 = (v: number) => {
    const b = Buffer.alloc(4);
    b.writeUInt32BE(v);
    return b;
  };
  // cmap: one (3,10) record pointing at a format 12 subtable.
  const groups = glyphs.map((g, i) => Buffer.concat([u32(g.cp), u32(g.cp), u32(i + 1)]));
  const sub = Buffer.concat([u16(12), u16(0), u32(16 + 12 * groups.length), u32(0), u32(groups.length), ...groups]);
  const cmap = Buffer.concat([u16(0), u16(1), u16(3), u16(10), u32(12), sub]);
  const maxp = Buffer.concat([u32(0x00005000), u16(numGlyphs)]);
  // sbix: one 160 ppem strike.
  const datas = [Buffer.alloc(0), ...glyphs.map((g) =>
    g.dupeOf !== undefined
      ? Buffer.concat([u16(0), u16(0), Buffer.from("dupe"), u16(g.dupeOf)])
      : Buffer.concat([u16(0), u16(0), Buffer.from("png "), g.png!]),
  )];
  const strikeHeader = 4 + 4 * (numGlyphs + 1);
  const offsets: number[] = [];
  let at = strikeHeader;
  for (const d of datas) {
    offsets.push(at);
    at += d.length;
  }
  offsets.push(at);
  const strike = Buffer.concat([u16(160), u16(72), ...offsets.map(u32), ...datas]);
  const sbix = Buffer.concat([u16(1), u16(1), u32(1), u32(12), strike]);
  const tables: [string, Buffer][] = [["cmap", cmap], ["maxp", maxp], ["sbix", sbix]];
  const fontStart = ttc ? 16 : 0;
  const dirSize = 12 + 16 * tables.length;
  let off = fontStart + dirSize;
  const records: Buffer[] = [];
  for (const [tag, data] of tables) {
    records.push(Buffer.concat([Buffer.from(tag), u32(0), u32(off), u32(data.length)]));
    off += data.length;
  }
  const font = Buffer.concat([u32(0x00010000), u16(tables.length), u16(0), u16(0), u16(0), ...records, ...tables.map((t) => t[1])]);
  return ttc ? Buffer.concat([Buffer.from("ttcf"), u32(0x00010000), u32(1), u32(16), font]) : font;
}

describe("sbix emoji font", () => {
  const lantern = solidPng(8, 8, [220, 30, 40]);
  const cup = solidPng(8, 8, [120, 70, 30]);

  it("maps a code point to its PNG bitmap, in a plain font and in a collection", () => {
    for (const ttc of [false, true]) {
      const font = openSbixFont(buildSbixFont([{ cp: 0x1f3ee, png: lantern }, { cp: 0x2615, png: cup }], ttc));
      expect(font.strikePpem).toBe(160);
      expect(font.png([0x1f3ee])!.equals(lantern)).toBe(true);
      expect(font.png([0x2615])!.equals(cup)).toBe(true);
    }
  });

  it("ignores a trailing emoji presentation selector and follows dupe glyphs", () => {
    const font = openSbixFont(buildSbixFont([{ cp: 0x2615, png: cup }, { cp: 0x2668, dupeOf: 1 }]));
    expect(font.png([0x2615, 0xfe0f])!.equals(cup)).toBe(true);
    expect(font.png([0x2668])!.equals(cup)).toBe(true);
  });

  it("returns nothing for sequences that need ligature lookups, and for unknown code points", () => {
    const font = openSbixFont(buildSbixFont([{ cp: 0x1f3ee, png: lantern }]));
    expect(font.png([0x1f469, 0x200d, 0x1f4bb])).toBeUndefined();
    expect(font.png([0x1f600])).toBeUndefined();
  });
});
