import { describe, expect, it } from "vitest";

import { parsePathData, rasterizeSvg } from "../src/svg-raster.js";

function alphaAt(img: { width: number; data: Uint8Array }, x: number, y: number): number {
  return img.data[(y * img.width + x) * 4 + 3]!;
}

function coverage(img: { width: number; height: number; data: Uint8Array }): number {
  let sum = 0;
  for (let i = 3; i < img.data.length; i += 4) sum += img.data[i]! / 255;
  return sum / (img.width * img.height);
}

describe("SVG path parsing", () => {
  it("reads compact numbers, implicit repeats and relative commands", () => {
    // "M10-5.5.5.5" is M 10,-5.5 then an implicit absolute lineto .5,.5.
    const sub = parsePathData("M10-5.5.5.5l5 0 0 5h-5zM0 0H4V4Z");
    expect(sub).toHaveLength(2);
    expect(sub[0]![0]).toEqual([10, -5.5]);
    expect(sub[0]![1]).toEqual([0.5, 0.5]);
    expect(sub[0]![2]).toEqual([5.5, 0.5]);
    expect(sub[1]!.at(-1)).toEqual([4, 4]);
  });

  it("reads arc flags written without separators", () => {
    const sub = parsePathData("M0 10a10 10 0 1120 0a10 10 0 11-20 0z");
    const xs = sub[0]!.map((p) => p[0]);
    expect(Math.min(...xs)).toBeCloseTo(0, 0);
    expect(Math.max(...xs)).toBeCloseTo(20, 0);
  });
});

describe("rasterizeSvg", () => {
  it("fills a square in viewBox coordinates, centred and scaled to the output", () => {
    const img = rasterizeSvg('<svg viewBox="0 -960 960 960"><path d="M0-960H960V0H0Z"/></svg>', 64, { padding: 0 });
    expect(img.width).toBe(64);
    expect(alphaAt(img, 0, 0)).toBe(255);
    expect(alphaAt(img, 63, 63)).toBe(255);
  });

  it("fills a circle made of arcs with about πr² coverage and leaves the corners clear", () => {
    const img = rasterizeSvg('<svg viewBox="0 0 20 20"><path d="M0 10a10 10 0 1 1 20 0a10 10 0 1 1 -20 0z"/></svg>', 128, { padding: 0 });
    expect(coverage(img)).toBeCloseTo(Math.PI / 4, 2);
    expect(alphaAt(img, 1, 1)).toBe(0);
    expect(alphaAt(img, 64, 64)).toBe(255);
  });

  it("cuts holes from sub-paths wound the other way (nonzero rule, as icon fonts use)", () => {
    const img = rasterizeSvg('<svg viewBox="0 0 10 10"><path d="M0 0H10V10H0ZM3 3V7H7V3Z"/></svg>', 100, { padding: 0 });
    expect(alphaAt(img, 5, 5)).toBe(255);
    expect(alphaAt(img, 50, 50)).toBe(0);
    expect(coverage(img)).toBeCloseTo(1 - 0.16, 2);
  });

  it("draws quadratic and cubic curves, with smooth continuations", () => {
    const img = rasterizeSvg('<svg viewBox="0 0 10 10"><path d="M0 10Q5 0 10 10T20 10ZM0 10C0 5 10 5 10 10S20 15 20 10Z"/></svg>', 64);
    expect(coverage(img)).toBeGreaterThan(0.05);
  });

  it("paints in the requested colour and adds padding around the viewBox", () => {
    const img = rasterizeSvg('<svg viewBox="0 0 10 10"><path d="M0 0H10V10H0Z"/></svg>', 100, { padding: 0.1, color: { r: 10, g: 20, b: 30 } });
    expect(alphaAt(img, 2, 2)).toBe(0);
    expect(Array.from(img.data.subarray((50 * 100 + 50) * 4, (50 * 100 + 50) * 4 + 4))).toEqual([10, 20, 30, 255]);
  });
});
