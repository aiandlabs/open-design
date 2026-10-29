import { describe, expect, it } from "vitest";

import {
  alphaStats,
  chromaKey,
  compositeOver,
  coverToSize,
  createImage,
  cutoutSheet,
  decodePng,
  decontaminateMatte,
  encodePng,
  estimateKeyColor,
  isPng,
  keyResidue,
  matteResidue,
  parseHexColor,
  pngSize,
  rectMask,
  referenceSimilarity,
  resize,
  toHexColor,
  type Rgb,
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

describe("cut-out residue", () => {
  const green = { r: 82, g: 198, b: 101 };

  it("reports a clean cut-out as clean", () => {
    const img = createImage(20, 20, { r: 0, g: 0, b: 0, a: 0 });
    for (let y = 5; y < 15; y += 1) for (let x = 5; x < 15; x += 1) img.data.set([220, 30, 40, 255], (y * 20 + x) * 4);
    expect(keyResidue(img, green)).toEqual({ keyTinted: 0, haze: 0 });
  });

  it("flags a translucent key-coloured shadow left beside the subject", () => {
    const img = createImage(20, 20, { r: 0, g: 0, b: 0, a: 0 });
    for (let y = 5; y < 15; y += 1) for (let x = 5; x < 10; x += 1) img.data.set([220, 30, 40, 255], (y * 20 + x) * 4);
    for (let y = 5; y < 15; y += 1) for (let x = 10; x < 15; x += 1) img.data.set([80, 190, 100, 120], (y * 20 + x) * 4);
    const r = keyResidue(img, green);
    expect(r.keyTinted).toBeCloseTo(0.5);
    expect(r.haze).toBeCloseTo(0.5);
  });

  it("ignores pixels that are effectively transparent", () => {
    const img = createImage(4, 4, { ...green, a: 8 });
    expect(keyResidue(img, green)).toEqual({ keyTinted: 0, haze: 0 });
  });

  it("lays the cut-out out on light and dark panels", () => {
    const img = createImage(64, 10, { r: 0, g: 0, b: 0, a: 0 });
    const sheet = cutoutSheet(img);
    expect(sheet.width).toBe(64 * 2 + 4);
    expect(sheet.height).toBe(10);
    expect(px(sheet, 0, 0)).toEqual([255, 255, 255, 255]);
    expect(px(sheet, 64 + 4, 0)).toEqual([24, 24, 24, 255]);
  });
});

describe("reference similarity (near-copy guard)", () => {
  /** A shape drawn into a transparent canvas: disc, square or a disc with a notch. */
  function shape(size: number, kind: "disc" | "square" | "notched", color: [number, number, number], offset = 0, scale = 1): RgbaImage {
    const img = createImage(size, size, { r: 0, g: 0, b: 0, a: 0 });
    const c = size / 2 + offset;
    const r = (size / 4) * scale;
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const inside =
          kind === "square" ? Math.abs(x - c) < r && Math.abs(y - c) < r
            : kind === "disc" ? Math.hypot(x - c, y - c) < r
              : Math.hypot(x - c, y - c) < r && !(x > c && y < c);
        // A dark stripe gives the shape internal structure for SSIM.
        if (inside) img.data.set(Math.abs(y - c) < r / 6 ? [20, 20, 20, 255] : [...color, 255], (y * size + x) * 4);
      }
    }
    return img;
  }

  it("scores a moved, scaled copy as a near-copy", () => {
    const s = referenceSimilarity(shape(128, "notched", [200, 30, 40], 10, 0.8), shape(128, "notched", [200, 30, 40]));
    expect(s.silhouetteIoU).toBeGreaterThan(0.95);
    expect(s.structure).toBeGreaterThan(0.9);
    expect(s.nearCopy).toBe(true);
  });

  it("does not flag a different shape", () => {
    const s = referenceSimilarity(shape(128, "square", [200, 30, 40]), shape(128, "notched", [200, 30, 40]));
    expect(s.silhouetteIoU).toBeLessThan(0.85);
    expect(s.nearCopy).toBe(false);
  });

  it("does not flag the same silhouette drawn with different internal structure", () => {
    const plain = shape(128, "disc", [200, 30, 40]);
    for (let i = 0; i < plain.data.length; i += 4) if (plain.data[i + 3]) plain.data.set([60, 140, 220], i);
    const s = referenceSimilarity(plain, shape(128, "disc", [200, 30, 40]));
    expect(s.silhouetteIoU).toBeGreaterThan(0.95);
    expect(s.nearCopy).toBe(false);
  });

  it("finds the subject of an opaque image by its difference from the border colour", () => {
    const opaque = compositeOver(shape(128, "notched", [200, 30, 40]), { r: 255, g: 255, b: 255 });
    expect(referenceSimilarity(opaque, shape(128, "notched", [200, 30, 40])).silhouetteIoU).toBeGreaterThan(0.95);
  });
});

describe("matte refinement and residue", () => {
  const grey = { r: 242, g: 242, b: 242 };
  const red = { r: 200, g: 30, b: 40 };

  /** Opaque red disc of radius r, matted from a grey backdrop, optionally with a ring of another colour. */
  function mattedDisc(size: number, r: number, ring?: { width: number; color: { r: number; g: number; b: number } }): RgbaImage {
    const img = createImage(size, size, { ...grey, a: 0 });
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const d = Math.hypot(x + 0.5 - size / 2, y + 0.5 - size / 2);
        const i = (y * size + x) * 4;
        if (d < r) img.data.set([red.r, red.g, red.b, 255], i);
        else if (ring && d < r + ring.width) img.data.set([ring.color.r, ring.color.g, ring.color.b, 255], i);
      }
    }
    return img;
  }

  it("takes the backdrop's colour back out of partly transparent edge pixels", () => {
    const img = createImage(2, 1, { r: 0, g: 0, b: 0, a: 0 });
    // Half-covered pixel: 0.5 × red + 0.5 × grey, as the matting model leaves it.
    img.data.set([221, 136, 141, 128], 0);
    img.data.set([10, 10, 10, 4], 4); // near-invisible speck
    const out = decontaminateMatte(img, grey);
    const [r, g, b, a] = px(out, 0, 0);
    expect(Math.abs(r! - red.r)).toBeLessThanOrEqual(2);
    expect(Math.abs(g! - red.g)).toBeLessThanOrEqual(2);
    expect(Math.abs(b! - red.b)).toBeLessThanOrEqual(2);
    expect(a).toBe(128);
    expect(px(out, 1, 0)[3]).toBe(0);
  });

  it("reports a clean matte as clean", () => {
    expect(matteResidue(mattedDisc(64, 20), grey)).toEqual({ fringe: 0, backdropTinted: 0, haze: 0 });
  });

  it("flags a backdrop-coloured fringe around the subject", () => {
    const r = matteResidue(mattedDisc(64, 20, { width: 2, color: { r: 236, g: 236, b: 236 } }), grey);
    expect(r.fringe).toBeGreaterThan(0.5);
    expect(r.backdropTinted).toBeGreaterThan(0.05);
  });

  it("does not count subject colours far from the backdrop as fringe", () => {
    const r = matteResidue(mattedDisc(64, 20, { width: 2, color: { r: 20, g: 20, b: 120 } }), grey);
    expect(r.fringe).toBe(0);
    expect(r.backdropTinted).toBe(0);
  });

  /** A disc whose own shading is backdrop-coloured on every third row, as a grey-ish subject on a grey backdrop. */
  function stripedDisc(ring?: { width: number; color: Rgb }) {
    const img = mattedDisc(96, 36, ring);
    for (let y = 0; y < 96; y += 3) {
      for (let x = 0; x < 96; x += 1) {
        const i = (y * 96 + x) * 4;
        if (Math.hypot(x + 0.5 - 48, y + 0.5 - 48) < 36) img.data.set([grey.r, grey.g, grey.b], i);
      }
    }
    return img;
  }

  it("does not flag a cleanly cut subject whose own colours are close to the backdrop", () => {
    const r = matteResidue(stripedDisc(), grey);
    expect(r.fringe).toBeLessThan(0.1);
    expect(r.backdropTinted).toBeLessThan(0.02);
  });

  it("still flags a backdrop fringe around a subject whose own colours are close to the backdrop", () => {
    const r = matteResidue(stripedDisc({ width: 3, color: grey }), grey);
    expect(r.fringe).toBeGreaterThan(0.1);
    expect(r.backdropTinted).toBeGreaterThan(0.02);
  });

  it("measures haze as the share of visible pixels that are only partly opaque", () => {
    const img = mattedDisc(64, 20);
    for (let i = 3; i < img.data.length; i += 4) if (img.data[i] === 0) img.data[i] = 60;
    expect(matteResidue(img, grey).haze).toBeGreaterThan(0.5);
  });
});
