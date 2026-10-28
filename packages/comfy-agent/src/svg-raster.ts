// A small SVG path rasteriser for single-colour icons (Material Symbols and
// similar icon-font SVGs): every <path> is filled with one colour using the
// nonzero rule, as icon fonts are drawn. Curves and arcs are flattened to line
// segments and filled at 4× supersampling. It does not implement strokes,
// transforms, gradients or CSS; icon references do not need them, and this
// keeps the package free of native image dependencies.

import { ImageFormatError } from "./errors.js";
import type { Rgb, RgbaImage } from "./image.js";

type Point = [number, number];

/**
 * Flattened sub-paths of an SVG path `d` attribute, in user units. Curves are
 * cut into chords about `step` user units long (use about one output pixel).
 */
export function parsePathData(d: string, step = 1): Point[][] {
  const segmentsFor = (length: number) => Math.max(4, Math.min(512, Math.ceil(length / step)));
  const subpaths: Point[][] = [];
  let cur: Point[] = [];
  let i = 0;
  let cmd = "";
  let x = 0;
  let y = 0;
  let startX = 0;
  let startY = 0;
  // Last control point, for S/s and T/t reflections.
  let ctrlX = 0;
  let ctrlY = 0;
  let lastCurve = "";

  const skip = () => {
    while (i < d.length && /[\s,]/.test(d[i]!)) i += 1;
  };
  const num = (): number => {
    skip();
    const m = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/.exec(d.slice(i));
    if (!m) throw new ImageFormatError(`invalid SVG path number at ${i}: "${d.slice(i, i + 12)}"`);
    i += m[0].length;
    return Number(m[0]);
  };
  const flag = (): number => {
    skip();
    const c = d[i];
    if (c !== "0" && c !== "1") throw new ImageFormatError(`invalid SVG arc flag at ${i}`);
    i += 1;
    return c === "1" ? 1 : 0;
  };
  const hasNumber = () => {
    skip();
    return i < d.length && /[0-9.+-]/.test(d[i]!);
  };
  const lineTo = (nx: number, ny: number) => {
    x = nx;
    y = ny;
    cur.push([x, y]);
  };
  const begin = (nx: number, ny: number) => {
    if (cur.length > 1) subpaths.push(cur);
    cur = [[nx, ny]];
    x = startX = nx;
    y = startY = ny;
  };

  while (true) {
    skip();
    if (i >= d.length) break;
    if (/[A-Za-z]/.test(d[i]!)) {
      cmd = d[i]!;
      i += 1;
    } else if (!cmd) {
      throw new ImageFormatError("SVG path data must start with a command");
    } else if (cmd === "M") {
      cmd = "L"; // pairs after a moveto are implicit linetos
    } else if (cmd === "m") {
      cmd = "l";
    }
    const rel = cmd === cmd.toLowerCase();
    const ox = rel ? x : 0;
    const oy = rel ? y : 0;
    switch (cmd.toUpperCase()) {
      case "M":
        begin(ox + num(), oy + num());
        lastCurve = "";
        break;
      case "L":
        lineTo(ox + num(), oy + num());
        lastCurve = "";
        break;
      case "H":
        lineTo((rel ? x : 0) + num(), y);
        lastCurve = "";
        break;
      case "V":
        lineTo(x, (rel ? y : 0) + num());
        lastCurve = "";
        break;
      case "C":
      case "S": {
        let x1: number;
        let y1: number;
        if (cmd.toUpperCase() === "C") {
          x1 = ox + num();
          y1 = oy + num();
        } else {
          x1 = lastCurve === "C" ? 2 * x - ctrlX : x;
          y1 = lastCurve === "C" ? 2 * y - ctrlY : y;
        }
        const x2 = ox + num();
        const y2 = oy + num();
        const ex = ox + num();
        const ey = oy + num();
        cubic(cur, segmentsFor, x, y, x1, y1, x2, y2, ex, ey);
        ctrlX = x2;
        ctrlY = y2;
        x = ex;
        y = ey;
        lastCurve = "C";
        break;
      }
      case "Q":
      case "T": {
        let x1: number;
        let y1: number;
        if (cmd.toUpperCase() === "Q") {
          x1 = ox + num();
          y1 = oy + num();
        } else {
          x1 = lastCurve === "Q" ? 2 * x - ctrlX : x;
          y1 = lastCurve === "Q" ? 2 * y - ctrlY : y;
        }
        const ex = ox + num();
        const ey = oy + num();
        quadratic(cur, segmentsFor, x, y, x1, y1, ex, ey);
        ctrlX = x1;
        ctrlY = y1;
        x = ex;
        y = ey;
        lastCurve = "Q";
        break;
      }
      case "A": {
        const rx = num();
        const ry = num();
        const rot = num();
        const large = flag();
        const sweep = flag();
        const ex = ox + num();
        const ey = oy + num();
        arc(cur, segmentsFor, x, y, rx, ry, rot, large, sweep, ex, ey);
        x = ex;
        y = ey;
        lastCurve = "";
        break;
      }
      case "Z":
        if (cur.length > 1) subpaths.push(cur);
        cur = [];
        x = startX;
        y = startY;
        cur = [[x, y]];
        lastCurve = "";
        // A Z takes no numbers; a following number would need a new command.
        if (hasNumber()) throw new ImageFormatError(`numbers after Z at ${i}`);
        cmd = "";
        break;
      default:
        throw new ImageFormatError(`unsupported SVG path command "${cmd}"`);
    }
  }
  if (cur.length > 1) subpaths.push(cur);
  return subpaths;
}

type Segments = (length: number) => number;

function quadratic(out: Point[], segmentsFor: Segments, x0: number, y0: number, x1: number, y1: number, x2: number, y2: number): void {
  const n = segmentsFor(Math.hypot(x1 - x0, y1 - y0) + Math.hypot(x2 - x1, y2 - y1));
  for (let k = 1; k <= n; k += 1) {
    const t = k / n;
    const u = 1 - t;
    out.push([u * u * x0 + 2 * u * t * x1 + t * t * x2, u * u * y0 + 2 * u * t * y1 + t * t * y2]);
  }
}

function cubic(out: Point[], segmentsFor: Segments, x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number): void {
  const n = segmentsFor(Math.hypot(x1 - x0, y1 - y0) + Math.hypot(x2 - x1, y2 - y1) + Math.hypot(x3 - x2, y3 - y2));
  for (let k = 1; k <= n; k += 1) {
    const t = k / n;
    const u = 1 - t;
    out.push([
      u * u * u * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t * x3,
      u * u * u * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y3,
    ]);
  }
}

/** Elliptical arc, SVG endpoint parameterisation (SVG 1.1 implementation notes, F.6.5). */
function arc(out: Point[], segmentsFor: Segments, x1: number, y1: number, rx: number, ry: number, rotDeg: number, large: number, sweep: number, x2: number, y2: number): void {
  if (x1 === x2 && y1 === y2) return;
  rx = Math.abs(rx);
  ry = Math.abs(ry);
  if (rx === 0 || ry === 0) {
    out.push([x2, y2]);
    return;
  }
  const phi = (rotDeg * Math.PI) / 180;
  const cos = Math.cos(phi);
  const sin = Math.sin(phi);
  const dx = (x1 - x2) / 2;
  const dy = (y1 - y2) / 2;
  const x1p = cos * dx + sin * dy;
  const y1p = -sin * dx + cos * dy;
  const lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
  if (lambda > 1) {
    rx *= Math.sqrt(lambda);
    ry *= Math.sqrt(lambda);
  }
  const num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
  const den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
  let coef = Math.sqrt(Math.max(0, num / den));
  if (large === sweep) coef = -coef;
  const cxp = (coef * rx * y1p) / ry;
  const cyp = (-coef * ry * x1p) / rx;
  const cx = cos * cxp - sin * cyp + (x1 + x2) / 2;
  const cy = sin * cxp + cos * cyp + (y1 + y2) / 2;
  const angle = (ux: number, uy: number, vx: number, vy: number) => {
    const a = Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy);
    return a;
  };
  const theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry);
  let delta = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry);
  if (!sweep && delta > 0) delta -= 2 * Math.PI;
  if (sweep && delta < 0) delta += 2 * Math.PI;
  const n = segmentsFor(Math.abs(delta) * Math.max(rx, ry));
  for (let k = 1; k <= n; k += 1) {
    const t = theta1 + (delta * k) / n;
    const ex = rx * Math.cos(t);
    const ey = ry * Math.sin(t);
    out.push([cos * ex - sin * ey + cx, sin * ex + cos * ey + cy]);
  }
}

export type RasterizeOptions = {
  /** Fill colour (default black). */
  color?: Rgb;
  /** Empty margin on each side, as a fraction of the output size (default 0.08). */
  padding?: number;
};

const SS = 4;

/**
 * Rasterise the `<path>` elements of a single-colour SVG to a `size`×`size`
 * RGBA image with a transparent background, fitted to the viewBox (centred,
 * aspect kept).
 */
export function rasterizeSvg(svg: string, size: number, options: RasterizeOptions = {}): RgbaImage {
  const vb = /viewBox\s*=\s*["']\s*([-\d.eE+]+)[\s,]+([-\d.eE+]+)[\s,]+([-\d.eE+]+)[\s,]+([-\d.eE+]+)\s*["']/.exec(svg);
  const w = /\bwidth\s*=\s*["']([\d.]+)/.exec(svg);
  const h = /\bheight\s*=\s*["']([\d.]+)/.exec(svg);
  const [minX, minY, vbW, vbH] = vb ? [Number(vb[1]), Number(vb[2]), Number(vb[3]), Number(vb[4])] : [0, 0, Number(w?.[1] ?? 24), Number(h?.[1] ?? 24)];
  if (!(vbW > 0 && vbH > 0)) throw new ImageFormatError("SVG has no usable viewBox or size");
  const paths = [...svg.matchAll(/<path\b[^>]*\bd\s*=\s*["']([^"']+)["']/g)].map((m) => m[1]!);
  if (paths.length === 0) throw new ImageFormatError("SVG has no <path> elements");

  const pad = (options.padding ?? 0.08) * size;
  const scale = Math.min((size - 2 * pad) / vbW, (size - 2 * pad) / vbH);
  const offX = (size - vbW * scale) / 2;
  const offY = (size - vbH * scale) / 2;
  const hi = size * SS;
  // Edges in supersampled pixel space.
  const edges: { x0: number; y0: number; x1: number; y1: number; dir: number }[] = [];
  for (const d of paths) {
    for (const sub of parsePathData(d, 1 / scale)) {
      const pts = sub.map(([px, py]) => [((px - minX) * scale + offX) * SS, ((py - minY) * scale + offY) * SS] as Point);
      for (let k = 0; k < pts.length; k += 1) {
        const [ax, ay] = pts[k]!;
        const [bx, by] = pts[(k + 1) % pts.length]!;
        if (ay === by) continue;
        edges.push(ay < by ? { x0: ax, y0: ay, x1: bx, y1: by, dir: 1 } : { x0: bx, y0: by, x1: ax, y1: ay, dir: -1 });
      }
    }
  }

  const cover = new Uint16Array(size * size);
  const crossings: { x: number; dir: number }[] = [];
  for (let sy = 0; sy < hi; sy += 1) {
    const yc = sy + 0.5;
    crossings.length = 0;
    for (const e of edges) {
      if (yc < e.y0 || yc >= e.y1) continue;
      crossings.push({ x: e.x0 + ((yc - e.y0) / (e.y1 - e.y0)) * (e.x1 - e.x0), dir: e.dir });
    }
    if (crossings.length < 2) continue;
    crossings.sort((a, b) => a.x - b.x);
    let winding = 0;
    const row = Math.floor(sy / SS) * size;
    for (let k = 0; k < crossings.length - 1; k += 1) {
      winding += crossings[k]!.dir;
      if (winding === 0) continue;
      const from = Math.max(0, Math.ceil(crossings[k]!.x - 0.5));
      const to = Math.min(hi - 1, Math.floor(crossings[k + 1]!.x - 0.5));
      for (let sx = from; sx <= to; sx += 1) cover[row + Math.floor(sx / SS)]! += 1;
    }
  }

  const color = options.color ?? { r: 0, g: 0, b: 0 };
  const data = new Uint8Array(size * size * 4);
  for (let p = 0; p < size * size; p += 1) {
    data[p * 4] = color.r;
    data[p * 4 + 1] = color.g;
    data[p * 4 + 2] = color.b;
    data[p * 4 + 3] = Math.round((cover[p]! / (SS * SS)) * 255);
  }
  return { width: size, height: size, data };
}
