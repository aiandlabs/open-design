import { describe, expect, it } from "vitest";

import {
  alphaStats,
  chromaKey,
  compositeOver,
  coverToSize,
  createImage,
  decodePng,
  encodePng,
  estimateKeyColor,
  isPng,
  parseHexColor,
  pngSize,
  rectMask,
  resize,
  toHexColor,
  type RgbaImage,
} from "../src/image.js";
import { ImageFormatError } from "../src/errors.js";

function px(img: RgbaImage, x: number, y: number) {
  const i = (y * img.width + x) * 4;
  return Array.from(img.data.subarray(i, i + 4));
}

/** Red disc (radius r) on a slightly noisy green background, like a real keyed render. */
function discOnGreen(w: number, h: number, r: number, withHole = false): RgbaImage {
  const img = createImage(w, h, { r: 0, g: 0, b: 0 });
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = (y * w + x) * 4;
      const d = Math.hypot(x + 0.5 - w / 2, y + 0.5 - h / 2);
      const inside = d < r;
      const hole = withHole && d < r / 4;
      if (inside && !hole) {
        img.data[i] = 210;
        img.data[i + 1] = 30;
        img.data[i + 2] = 40;
      } else {
        img.data[i] = 20 + ((x * 7 + y * 3) % 9);
        img.data[i + 1] = 235 - ((x + y) % 7);
        img.data[i + 2] = 30;
      }
      img.data[i + 3] = 255;
    }
  }
  return img;
}

describe("PNG I/O", () => {
  it("round-trips and reads the header", () => {
    const img = createImage(7, 5, { r: 1, g: 2, b: 3, a: 128 });
    const png = encodePng(img);
    expect(isPng(png)).toBe(true);
    expect(pngSize(png)).toEqual({ width: 7, height: 5, hasAlphaChannel: true });
    const back = decodePng(png);
    expect(back.width).toBe(7);
    expect(px(back, 3, 2)).toEqual([1, 2, 3, 128]);
  });

  it("rejects non-PNG bytes", () => {
    expect(() => decodePng(Buffer.from("not an image"))).toThrow(ImageFormatError);
  });
});

describe("resize and crop", () => {
  it("resizes to exact dimensions and preserves flat colour", () => {
    const out = resize(createImage(100, 60, { r: 10, g: 200, b: 30 }), 37, 91);
    expect([out.width, out.height]).toEqual([37, 91]);
    expect(px(out, 18, 45)).toEqual([10, 200, 30, 255]);
  });

  it("does not bleed transparent pixels' colour into edges", () => {
    const img = createImage(4, 1, { r: 255, g: 255, b: 255, a: 255 });
    // Right half transparent black.
    for (const x of [2, 3]) img.data.set([0, 0, 0, 0], x * 4);
    const out = resize(img, 2, 1);
    expect(px(out, 1, 0)[3]).toBeLessThan(128);
    // Where visible, the colour stays white rather than greying toward black.
    expect(px(out, 0, 0).slice(0, 3)).toEqual([255, 255, 255]);
  });

  it("cover-crops to the requested size without distortion", () => {
    // Left third blue, rest red; cropping a wide image to square keeps the centre.
    const img = createImage(300, 100, { r: 255, g: 0, b: 0 });
    for (let y = 0; y < 100; y += 1) for (let x = 0; x < 100; x += 1) img.data.set([0, 0, 255, 255], (y * 300 + x) * 4);
    const out = coverToSize(img, 50, 50);
    expect([out.width, out.height]).toEqual([50, 50]);
    expect(px(out, 25, 25)).toEqual([255, 0, 0, 255]);
    const bigger = coverToSize(createImage(768, 768, { r: 9, g: 9, b: 9 }), 1000, 500);
    expect([bigger.width, bigger.height]).toEqual([1000, 500]);
  });
});

describe("chroma key", () => {
  it("estimates the rendered key colour from the border", () => {
    const img = discOnGreen(64, 64, 20);
    const est = estimateKeyColor(img, parseHexColor("#00FF00"));
    expect(est.borderMatch).toBe(1);
    expect(est.color.g).toBeGreaterThan(225);
    expect(est.color.r).toBeLessThan(40);
  });

  it("keys the background to alpha with a feathered edge and keeps the subject opaque", () => {
    const img = discOnGreen(64, 64, 20);
    const est = estimateKeyColor(img, parseHexColor("#00FF00"));
    const out = chromaKey(img, est.color, { featherPx: 1 });
    expect(px(out, 0, 0)[3]).toBe(0);
    expect(px(out, 32, 32)).toEqual([210, 30, 40, 255]);
    const stats = alphaStats(out);
    expect(stats.borderTransparent).toBe(1);
    expect(stats.transparent).toBeGreaterThan(0.5);
    // Feathering produces partial alpha on the edge ring.
    const edgeAlphas: number[] = [];
    for (let x = 0; x < 64; x += 1) edgeAlphas.push(px(out, x, 32)[3]!);
    expect(edgeAlphas.some((a) => a > 0 && a < 255)).toBe(true);
  });

  it("despills key colour from semi-transparent edge pixels", () => {
    // A pixel that is a 50/50 mix of red subject and green key.
    const img = createImage(3, 3, { r: 0, g: 255, b: 0 });
    img.data.set([128, 128, 20, 255], 4 * 4);
    const out = chromaKey(img, { r: 0, g: 255, b: 0 }, { tolerance: 0.05, softness: 0.6, featherPx: 0, connectedOnly: false });
    const [r, g, , a] = px(out, 1, 1);
    expect(a).toBeGreaterThan(0);
    expect(a).toBeLessThan(255);
    expect(g!).toBeLessThan(128);
    expect(r!).toBeGreaterThanOrEqual(128);
  });

  it("keeps key-coloured regions that are not connected to the border", () => {
    const img = discOnGreen(64, 64, 24, true);
    const out = chromaKey(img, estimateKeyColor(img, parseHexColor("#00FF00")).color, { connectedOnly: true, featherPx: 0 });
    expect(px(out, 32, 32)[3]).toBe(255);
    const global = chromaKey(img, estimateKeyColor(img, parseHexColor("#00FF00")).color, { connectedOnly: false, featherPx: 0 });
    expect(px(global, 32, 32)[3]).toBe(0);
  });

  it("composites cut-outs over a flat colour", () => {
    const img = createImage(2, 1, { r: 255, g: 0, b: 0, a: 0 });
    expect(px(compositeOver(img, { r: 128, g: 128, b: 128 }), 0, 0)).toEqual([128, 128, 128, 255]);
  });
});

describe("masks and colours", () => {
  it("draws a feathered rectangular mask from fractional coordinates", () => {
    const m = rectMask(100, 100, { x: 0.25, y: 0.25, width: 0.5, height: 0.5 }, 4);
    expect(px(m, 50, 50)[0]).toBe(255);
    expect(px(m, 5, 5)[0]).toBe(0);
    const edge = px(m, 23, 50)[0]!;
    expect(edge).toBeGreaterThan(0);
    expect(edge).toBeLessThan(255);
  });

  it("parses and formats hex colours", () => {
    expect(parseHexColor("#ff00FF")).toEqual({ r: 255, g: 0, b: 255 });
    expect(toHexColor({ r: 0, g: 255, b: 16 })).toBe("#00ff10");
    expect(() => parseHexColor("green")).toThrow(ImageFormatError);
  });
});
