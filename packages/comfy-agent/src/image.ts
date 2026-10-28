// Deterministic image post-processing done in code, not by a model.
//
// PNG I/O uses `pngjs` (pure JavaScript, no native addon). The desktop app is
// packaged per platform and architecture; a native image library such as
// sharp ships prebuilt libvips binaries per target, has to survive
// electron-rebuild and asar unpacking, and adds tens of megabytes. Assets
// here are at most a few megapixels and are processed once per round, so a
// pure-JS codec is fast enough and keeps the package architecture-neutral.
// Resampling and keying are implemented below on raw RGBA buffers.

import { PNG } from "pngjs";

import { ImageFormatError } from "./errors.js";

export type RgbaImage = { width: number; height: number; data: Uint8Array };
export type Rgb = { r: number; g: number; b: number };

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

export function isPng(bytes: Uint8Array): boolean {
  return bytes.length > 8 && PNG_SIGNATURE.every((b, i) => bytes[i] === b);
}

export function decodePng(bytes: Uint8Array): RgbaImage {
  if (!isPng(bytes)) throw new ImageFormatError("expected PNG bytes (ComfyUI SaveImage writes PNG)");
  try {
    const png = PNG.sync.read(Buffer.from(bytes));
    return { width: png.width, height: png.height, data: new Uint8Array(png.data.buffer, png.data.byteOffset, png.data.byteLength) };
  } catch (err) {
    throw new ImageFormatError(`could not decode PNG: ${err instanceof Error ? err.message : String(err)}`, err);
  }
}

export function encodePng(img: RgbaImage): Buffer {
  const png = new PNG({ width: img.width, height: img.height });
  png.data = Buffer.from(img.data.buffer, img.data.byteOffset, img.data.byteLength);
  return PNG.sync.write(png, { colorType: 6 });
}

/** Read the IHDR without decoding the whole image. */
export function pngSize(bytes: Uint8Array): { width: number; height: number; hasAlphaChannel: boolean } {
  if (!isPng(bytes) || bytes.length < 26) throw new ImageFormatError("not a PNG");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const colorType = bytes[25]!;
  return { width: view.getUint32(16), height: view.getUint32(20), hasAlphaChannel: colorType === 4 || colorType === 6 };
}

export function createImage(width: number, height: number, fill: Rgb & { a?: number } = { r: 0, g: 0, b: 0, a: 255 }): RgbaImage {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = fill.r;
    data[i + 1] = fill.g;
    data[i + 2] = fill.b;
    data[i + 3] = fill.a ?? 255;
  }
  return { width, height, data };
}

// ---------------------------------------------------------------------------
// Resampling: separable triangle filter whose support widens with the
// downscale factor (area-averaging when shrinking, bilinear when growing).
// Colour is premultiplied by alpha while filtering so transparent pixels do
// not bleed their RGB into the edges.

function resampleAxis(
  src: Float32Array,
  srcW: number,
  srcH: number,
  dstLen: number,
  horizontal: boolean,
): Float32Array {
  const srcLen = horizontal ? srcW : srcH;
  const other = horizontal ? srcH : srcW;
  const scale = srcLen / dstLen;
  const support = Math.max(1, scale);
  const out = new Float32Array((horizontal ? dstLen * srcH : srcW * dstLen) * 4);
  for (let d = 0; d < dstLen; d += 1) {
    const center = (d + 0.5) * scale;
    const lo = Math.max(0, Math.floor(center - support));
    const hi = Math.min(srcLen - 1, Math.ceil(center + support));
    const weights: number[] = [];
    let total = 0;
    for (let s = lo; s <= hi; s += 1) {
      const w = Math.max(0, 1 - Math.abs((s + 0.5 - center) / support));
      weights.push(w);
      total += w;
    }
    if (total === 0) {
      weights.fill(0);
      weights[Math.min(weights.length - 1, Math.max(0, Math.round(center - 0.5) - lo))] = 1;
      total = 1;
    }
    for (let o = 0; o < other; o += 1) {
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;
      for (let k = 0; k < weights.length; k += 1) {
        const w = weights[k]! / total;
        if (w === 0) continue;
        const s = lo + k;
        const idx = horizontal ? (o * srcW + s) * 4 : (s * srcW + o) * 4;
        r += src[idx]! * w;
        g += src[idx + 1]! * w;
        b += src[idx + 2]! * w;
        a += src[idx + 3]! * w;
      }
      const di = horizontal ? (o * dstLen + d) * 4 : (d * srcW + o) * 4;
      out[di] = r;
      out[di + 1] = g;
      out[di + 2] = b;
      out[di + 3] = a;
    }
  }
  return out;
}

export function resize(img: RgbaImage, width: number, height: number): RgbaImage {
  if (width <= 0 || height <= 0) throw new ImageFormatError(`invalid target size ${width}x${height}`);
  if (width === img.width && height === img.height) return { ...img, data: new Uint8Array(img.data) };
  const n = img.width * img.height;
  const pre = new Float32Array(n * 4);
  for (let i = 0; i < n * 4; i += 4) {
    const a = img.data[i + 3]! / 255;
    pre[i] = img.data[i]! * a;
    pre[i + 1] = img.data[i + 1]! * a;
    pre[i + 2] = img.data[i + 2]! * a;
    pre[i + 3] = img.data[i + 3]!;
  }
  const horiz = resampleAxis(pre, img.width, img.height, width, true);
  const both = resampleAxis(horiz, width, img.height, height, false);
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    const a = both[i + 3]!;
    const inv = a > 0 ? 255 / a : 0;
    data[i] = clamp255(both[i]! * inv);
    data[i + 1] = clamp255(both[i + 1]! * inv);
    data[i + 2] = clamp255(both[i + 2]! * inv);
    data[i + 3] = clamp255(a);
  }
  return { width, height, data };
}

export function crop(img: RgbaImage, x: number, y: number, width: number, height: number): RgbaImage {
  if (x < 0 || y < 0 || x + width > img.width || y + height > img.height) {
    throw new ImageFormatError(`crop ${x},${y} ${width}x${height} is outside ${img.width}x${img.height}`);
  }
  const data = new Uint8Array(width * height * 4);
  for (let row = 0; row < height; row += 1) {
    const from = ((y + row) * img.width + x) * 4;
    data.set(img.data.subarray(from, from + width * 4), row * width * 4);
  }
  return { width, height, data };
}

/**
 * Scale to cover the target box, then centre-crop to exactly width × height.
 * This is how a generation at a native bucket size becomes the requested size
 * without distorting the aspect ratio.
 */
export function coverToSize(img: RgbaImage, width: number, height: number): RgbaImage {
  if (img.width === width && img.height === height) return img;
  const scale = Math.max(width / img.width, height / img.height);
  const scaledW = Math.max(width, Math.round(img.width * scale));
  const scaledH = Math.max(height, Math.round(img.height * scale));
  const scaled = resize(img, scaledW, scaledH);
  return crop(scaled, Math.floor((scaledW - width) / 2), Math.floor((scaledH - height) / 2), width, height);
}

// ---------------------------------------------------------------------------
// Alpha inspection

export type AlphaStats = {
  /** Fraction of pixels with alpha < 255. */
  translucent: number;
  /** Fraction of pixels with alpha === 0. */
  transparent: number;
  /** Fraction of border pixels (1px frame) with alpha < 32. */
  borderTransparent: number;
};

export function alphaStats(img: RgbaImage): AlphaStats {
  const n = img.width * img.height;
  let translucent = 0;
  let transparent = 0;
  for (let i = 3; i < img.data.length; i += 4) {
    const a = img.data[i]!;
    if (a < 255) translucent += 1;
    if (a === 0) transparent += 1;
  }
  let border = 0;
  let borderClear = 0;
  forEachBorderPixel(img, (idx) => {
    border += 1;
    if (img.data[idx + 3]! < 32) borderClear += 1;
  });
  return { translucent: translucent / n, transparent: transparent / n, borderTransparent: border ? borderClear / border : 0 };
}

function forEachBorderPixel(img: RgbaImage, fn: (idx: number, x: number, y: number) => void): void {
  const { width: w, height: h } = img;
  for (let x = 0; x < w; x += 1) {
    fn(x * 4, x, 0);
    if (h > 1) fn(((h - 1) * w + x) * 4, x, h - 1);
  }
  for (let y = 1; y < h - 1; y += 1) {
    fn(y * w * 4, 0, y);
    if (w > 1) fn((y * w + w - 1) * 4, w - 1, y);
  }
}

// ---------------------------------------------------------------------------
// Chroma key

export function parseHexColor(hex: string): Rgb {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) throw new ImageFormatError(`invalid hex colour "${hex}"; expected #RRGGBB`);
  const v = Number.parseInt(m[1]!, 16);
  return { r: (v >> 16) & 255, g: (v >> 8) & 255, b: v & 255 };
}

export function toHexColor(c: Rgb): string {
  return `#${[c.r, c.g, c.b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
}

/** Normalised colour distance in [0, 1]. */
export function colorDistance(a: Rgb, b: Rgb): number {
  const dr = a.r - b.r;
  const dg = a.g - b.g;
  const db = a.b - b.b;
  // Weighted toward perceived differences (redmean approximation).
  const rmean = (a.r + b.r) / 2;
  const d = Math.sqrt((2 + rmean / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rmean) / 256) * db * db);
  return d / 764.8; // max distance (black ↔ white) ≈ 764.8
}

/**
 * Diffusion models never paint the exact key colour. Estimate the colour the
 * background actually came out as: the median of border pixels that are
 * reasonably close to the requested key (or of all border pixels when none are).
 */
export function estimateKeyColor(img: RgbaImage, nominal: Rgb, maxDistance = 0.35): { color: Rgb; borderMatch: number } {
  const near: Rgb[] = [];
  const all: Rgb[] = [];
  forEachBorderPixel(img, (idx) => {
    const c = { r: img.data[idx]!, g: img.data[idx + 1]!, b: img.data[idx + 2]! };
    all.push(c);
    if (colorDistance(c, nominal) <= maxDistance) near.push(c);
  });
  const pool = near.length >= Math.max(8, all.length * 0.1) ? near : all;
  const median = (vals: number[]) => {
    const s = vals.slice().sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)] ?? 0;
  };
  return {
    color: { r: median(pool.map((c) => c.r)), g: median(pool.map((c) => c.g)), b: median(pool.map((c) => c.b)) },
    borderMatch: all.length ? near.length / all.length : 0,
  };
}

export type ChromaKeyOptions = {
  /** Distance below which a pixel is fully transparent. */
  tolerance?: number;
  /** Width of the soft ramp above `tolerance`. */
  softness?: number;
  /** Only remove key-coloured regions connected to the image border (keeps interior details). */
  connectedOnly?: boolean;
  /** Alpha edge feather radius in pixels (box blur on the alpha channel). */
  featherPx?: number;
  /** Remove key-colour spill from semi-transparent edge pixels. */
  despill?: boolean;
};

export function chromaKey(img: RgbaImage, key: Rgb, options: ChromaKeyOptions = {}): RgbaImage {
  const tolerance = options.tolerance ?? 0.12;
  const softness = options.softness ?? 0.1;
  const connectedOnly = options.connectedOnly ?? true;
  const featherPx = options.featherPx ?? 1;
  const despill = options.despill ?? true;
  const { width: w, height: h } = img;
  const n = w * h;
  const dist = new Float32Array(n);
  for (let p = 0; p < n; p += 1) {
    const i = p * 4;
    dist[p] = colorDistance({ r: img.data[i]!, g: img.data[i + 1]!, b: img.data[i + 2]! }, key);
  }
  const keyish = (p: number) => dist[p]! < tolerance + softness;

  // Which key-like pixels are reachable from the border.
  let reachable: Uint8Array | null = null;
  if (connectedOnly) {
    reachable = new Uint8Array(n);
    const stack: number[] = [];
    const push = (p: number) => {
      if (!reachable![p] && keyish(p)) {
        reachable![p] = 1;
        stack.push(p);
      }
    };
    for (let x = 0; x < w; x += 1) {
      push(x);
      push((h - 1) * w + x);
    }
    for (let y = 0; y < h; y += 1) {
      push(y * w);
      push(y * w + w - 1);
    }
    while (stack.length > 0) {
      const p = stack.pop()!;
      const x = p % w;
      const y = (p - x) / w;
      if (x > 0) push(p - 1);
      if (x < w - 1) push(p + 1);
      if (y > 0) push(p - w);
      if (y < h - 1) push(p + w);
    }
  }

  let alpha: Float32Array = new Float32Array(n);
  for (let p = 0; p < n; p += 1) {
    const original = img.data[p * 4 + 3]! / 255;
    if (reachable && !reachable[p]) {
      alpha[p] = original;
      continue;
    }
    const t = (dist[p]! - tolerance) / softness;
    const ramp = t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
    alpha[p] = Math.min(original, ramp);
  }

  if (featherPx > 0) alpha = boxBlurAlpha(alpha, w, h, featherPx);

  const data = new Uint8Array(img.data);
  for (let p = 0; p < n; p += 1) {
    const i = p * 4;
    const a = alpha[p]!;
    data[i + 3] = clamp255(a * 255);
    if (despill && a > 0.02 && a < 0.999) {
      // Un-mix the key colour: observed = a * fg + (1 - a) * key.
      data[i] = clamp255((img.data[i]! - (1 - a) * key.r) / a);
      data[i + 1] = clamp255((img.data[i + 1]! - (1 - a) * key.g) / a);
      data[i + 2] = clamp255((img.data[i + 2]! - (1 - a) * key.b) / a);
    }
    if (a <= 0.02) {
      data[i] = 0;
      data[i + 1] = 0;
      data[i + 2] = 0;
      data[i + 3] = 0;
    }
  }
  return { width: w, height: h, data };
}

function boxBlurAlpha(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  const r = Math.max(1, Math.round(radius));
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      let count = 0;
      for (let k = -r; k <= r; k += 1) {
        const xx = x + k;
        if (xx < 0 || xx >= w) continue;
        sum += src[y * w + xx]!;
        count += 1;
      }
      tmp[y * w + x] = sum / count;
    }
  }
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let sum = 0;
      let count = 0;
      for (let k = -r; k <= r; k += 1) {
        const yy = y + k;
        if (yy < 0 || yy >= h) continue;
        sum += tmp[yy * w + x]!;
        count += 1;
      }
      // Blur only softens edges; it never makes a fully opaque interior translucent
      // or reintroduces opacity deep in the keyed background.
      const blurred = sum / count;
      const orig = src[y * w + x]!;
      out[y * w + x] = orig === 1 && blurred > 0.999 ? 1 : orig === 0 && blurred < 0.001 ? 0 : blurred;
    }
  }
  return out;
}

/** Composite onto a flat colour (used to show a cut-out to a vision model). */
export function compositeOver(img: RgbaImage, bg: Rgb): RgbaImage {
  const data = new Uint8Array(img.data.length);
  for (let i = 0; i < data.length; i += 4) {
    const a = img.data[i + 3]! / 255;
    data[i] = clamp255(img.data[i]! * a + bg.r * (1 - a));
    data[i + 1] = clamp255(img.data[i + 1]! * a + bg.g * (1 - a));
    data[i + 2] = clamp255(img.data[i + 2]! * a + bg.b * (1 - a));
    data[i + 3] = 255;
  }
  return { width: img.width, height: img.height, data };
}

/**
 * Show a cut-out to a vision model on a light and a dark background, side by
 * side. Residue from keying (tinted shadows, haze, halos) is easy to miss on a
 * single mid-grey background but stands out against at least one of these.
 */
export function cutoutSheet(img: RgbaImage, light: Rgb = { r: 255, g: 255, b: 255 }, dark: Rgb = { r: 24, g: 24, b: 24 }): RgbaImage {
  const gap = Math.max(4, Math.round(img.width / 64));
  const sheet = createImage(img.width * 2 + gap, img.height, { r: 255, g: 0, b: 255, a: 255 });
  for (const [i, panel] of [compositeOver(img, light), compositeOver(img, dark)].entries()) {
    const x0 = i * (img.width + gap);
    for (let y = 0; y < img.height; y += 1) {
      sheet.data.set(panel.data.subarray(y * img.width * 4, (y + 1) * img.width * 4), (y * sheet.width + x0) * 4);
    }
  }
  return sheet;
}

export type KeyResidue = {
  /** Fraction of visible pixels (alpha ≥ 16) whose colour is still near the key. */
  keyTinted: number;
  /** Fraction of visible pixels that are partly transparent (16 ≤ alpha < 224). */
  haze: number;
};

/**
 * Measure what chroma keying left behind. A clean cut-out has almost no
 * visible pixels near the key colour, and partial alpha only in a thin edge
 * band; a model-painted drop shadow on the key colour shows up in both.
 */
export function keyResidue(img: RgbaImage, key: Rgb, maxDistance = 0.22): KeyResidue {
  let visible = 0;
  let tinted = 0;
  let partial = 0;
  for (let i = 0; i < img.data.length; i += 4) {
    const a = img.data[i + 3]!;
    if (a < 16) continue;
    visible += 1;
    if (a < 224) partial += 1;
    if (colorDistance({ r: img.data[i]!, g: img.data[i + 1]!, b: img.data[i + 2]! }, key) <= maxDistance) tinted += 1;
  }
  return visible ? { keyTinted: tinted / visible, haze: partial / visible } : { keyTinted: 0, haze: 0 };
}

export type NormalizedRect = { x: number; y: number; width: number; height: number };

/**
 * Inpaint mask: white (255) inside the rectangle, black outside, with a
 * linear feather band. Coordinates are fractions of the image size.
 */
export function rectMask(width: number, height: number, rect: NormalizedRect, featherPx = 8): RgbaImage {
  const x0 = Math.max(0, Math.min(width, rect.x * width));
  const y0 = Math.max(0, Math.min(height, rect.y * height));
  const x1 = Math.max(x0, Math.min(width, (rect.x + rect.width) * width));
  const y1 = Math.max(y0, Math.min(height, (rect.y + rect.height) * height));
  const img = createImage(width, height, { r: 0, g: 0, b: 0, a: 255 });
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = Math.max(x0 - (x + 0.5), x + 0.5 - x1, 0);
      const dy = Math.max(y0 - (y + 0.5), y + 0.5 - y1, 0);
      const d = Math.max(dx, dy);
      const v = featherPx > 0 ? Math.max(0, 1 - d / featherPx) : d === 0 ? 1 : 0;
      const i = (y * width + x) * 4;
      const c = clamp255(v * 255);
      img.data[i] = c;
      img.data[i + 1] = c;
      img.data[i + 2] = c;
    }
  }
  return img;
}

/** Shrink so the long side is at most `maxSide` (for vision-model payloads). */
export function fitWithin(img: RgbaImage, maxSide: number): RgbaImage {
  const long = Math.max(img.width, img.height);
  if (long <= maxSide) return img;
  const s = maxSide / long;
  return resize(img, Math.max(1, Math.round(img.width * s)), Math.max(1, Math.round(img.height * s)));
}

export function pngDataUrl(bytes: Uint8Array): string {
  return `data:image/png;base64,${Buffer.from(bytes).toString("base64")}`;
}

/** Most common coarse colours (for telling the planner what the subject looks like). */
export function dominantColors(img: RgbaImage, count = 5): { hex: string; share: number }[] {
  const buckets = new Map<number, { r: number; g: number; b: number; n: number }>();
  let total = 0;
  for (let i = 0; i < img.data.length; i += 16) {
    if (img.data[i + 3]! < 128) continue;
    const r = img.data[i]!;
    const g = img.data[i + 1]!;
    const b = img.data[i + 2]!;
    const key = ((r >> 5) << 6) | ((g >> 5) << 3) | (b >> 5);
    const cur = buckets.get(key) ?? { r: 0, g: 0, b: 0, n: 0 };
    cur.r += r;
    cur.g += g;
    cur.b += b;
    cur.n += 1;
    buckets.set(key, cur);
    total += 1;
  }
  return Array.from(buckets.values())
    .sort((a, b) => b.n - a.n)
    .slice(0, count)
    .map((c) => ({ hex: toHexColor({ r: c.r / c.n, g: c.g / c.n, b: c.b / c.n }), share: total ? c.n / total : 0 }));
}

function clamp255(v: number): number {
  return v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v);
}
