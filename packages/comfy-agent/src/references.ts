// Shape and concept references for icon generation: the closest emoji and
// Material Symbols for a request, rendered as transparent PNGs. A reference
// only conditions the image model (it supplies shape and concept, never the
// look) and is not placed in a design, so sources are chosen for coverage and
// drawing quality. Every reference records where it came from. Nothing third
// party is bundled: indexes and images are fetched from pinned versions and
// cached under the caller's cache directory, and the system emoji font is read
// in place when the machine has one.

import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { openSbixFont, type SbixFont } from "./emoji-font.js";
import { ComfyAgentError } from "./errors.js";
import { createImage, decodePng, encodePng, resize, type RgbaImage } from "./image.js";
import { rasterizeSvg } from "./svg-raster.js";

/** Pinned reference sources. Changing one changes which references a request gets. */
export const REFERENCE_SOURCES = {
  emojiIndex: {
    label: "emojibase-data 17.0.0 (CLDR names and keywords)",
    url: "https://cdn.jsdelivr.net/npm/emojibase-data@17.0.0/en/compact.json",
  },
  materialIndex: {
    label: "Google Fonts icon metadata (Material Symbols names and tags)",
    url: "https://fonts.google.com/metadata/icons?key=material_symbols&incomplete=true",
  },
  materialSvg: {
    label: "Material Symbols 0.48.2 (@material-symbols/svg-400, outlined, filled)",
    url: (name: string, filled: boolean) => `https://cdn.jsdelivr.net/npm/@material-symbols/svg-400@0.48.2/outlined/${name}${filled ? "-fill" : ""}.svg`,
  },
  appleEmoji: { label: "Apple Color Emoji (system font)", path: "/System/Library/Fonts/Apple Color Emoji.ttc" },
  fluentEmoji: {
    label: "Fluent Emoji 3D (microsoft/fluentui-emoji@1ffb34c)",
    url: (folder: string, file: string) =>
      `https://cdn.jsdelivr.net/gh/microsoft/fluentui-emoji@1ffb34c752ecf5d402f04cfb4b392c77f57c54bc/assets/${encodeURIComponent(folder)}/${file}`,
  },
  notoEmoji: {
    label: "Noto Emoji v2.051",
    url: (file: string) => `https://cdn.jsdelivr.net/gh/googlefonts/noto-emoji@v2.051/png/512/${file}`,
  },
} as const;

export type EmojiRenderer = "apple" | "fluent" | "noto";

export type ReferenceCandidate = {
  /** `emoji:<hexcode>` or `icon:<material symbol name>`. */
  id: string;
  kind: "emoji" | "icon";
  label: string;
  tags: string[];
  score: number;
};

export type ReferenceAsset = {
  id: string;
  kind: "emoji" | "icon";
  label: string;
  /** Where the pixels came from (set and version). */
  source: string;
  url?: string;
  /** Transparent PNG of the subject. */
  png: Buffer;
};

export type ReferenceLibraryOptions = {
  /** Where indexes and fetched images are cached. */
  cacheDir: string;
  fetch?: typeof fetch;
  /** Emoji image sources, tried in order (default apple, fluent, noto). */
  emojiRenderers?: EmojiRenderer[];
  /** Path of an sbix colour emoji font; false to never read one (default: the macOS system font when present). */
  appleEmojiFont?: string | false;
  timeoutMs?: number;
};

type EmojiEntry = { hexcode: string; label: string; tags?: string[]; group?: number };
type IconEntry = { name: string; tags: string[]; categories?: string[] };

const STOPWORDS = new Set(
  (
    "a an the and or of with without in on for to at by from into icon icons emoji symbol logo flat style simple single one " +
    "minimal minimalist modern vector illustration illustrated cartoon cute friendly small large big little round rounded " +
    "centred centered transparent background white black colour color colours colors coloured colored palette brand glossy " +
    "3d app ui design set family matching consistent outline outlined filled line thin thick bold soft bright dark light " +
    "red orange yellow green blue purple pink brown grey gray cream gold silver teal navy beige pastel up down left right " +
    "pointing facing showing shows its it is that this has have"
  ).split(/\s+/),
);

/** Content words of a request, lower case and singular, for reference search. */
export function referenceQueryTerms(text: string): string[] {
  const words = text.toLowerCase().normalize("NFKD").replace(/[^a-z0-9\s-]/g, " ").split(/[\s-]+/);
  const out: string[] = [];
  for (const w of words) {
    if (w.length < 2 || STOPWORDS.has(w) || /^\d+$/.test(w)) continue;
    const s = singular(w);
    if (!out.includes(s)) out.push(s);
  }
  return out;
}

function singular(w: string): string {
  if (w.length > 4 && w.endsWith("ies")) return `${w.slice(0, -3)}y`;
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss") && !w.endsWith("us")) return w.slice(0, -1);
  return w;
}

/**
 * Relevance of one candidate: request terms found in its name count most,
 * terms found in its tags less; name words the request does not mention cost
 * a little, so "coffee" prefers "coffee" over "coffee maker".
 */
export function scoreCandidate(terms: string[], label: string, tags: string[]): number {
  if (terms.length === 0) return 0;
  const labelWords = referenceQueryTerms(label);
  const tagWords = new Set(tags.flatMap((t) => referenceQueryTerms(t)));
  let score = 0;
  let matchedLabel = 0;
  for (const t of terms) {
    if (labelWords.includes(t)) {
      score += 3;
      matchedLabel += 1;
    } else if (tagWords.has(t)) {
      score += 1.5;
    }
  }
  if (score === 0) return 0;
  return score - 0.4 * Math.max(0, labelWords.length - matchedLabel);
}

export class ReferenceLibrary {
  private readonly cacheDir: string;
  private readonly fetchImpl: typeof fetch;
  private readonly renderers: EmojiRenderer[];
  private readonly fontPath: string | undefined;
  private readonly timeoutMs: number;
  private emojiIndex?: Promise<EmojiEntry[]>;
  private iconIndex?: Promise<IconEntry[]>;
  private font?: SbixFont | null;
  /** sha256 of each index as read, for the trace. */
  readonly indexDigests: Record<string, string> = {};

  constructor(options: ReferenceLibraryOptions) {
    this.cacheDir = options.cacheDir;
    this.fetchImpl = options.fetch ?? fetch;
    this.renderers = options.emojiRenderers ?? ["apple", "fluent", "noto"];
    const font = options.appleEmojiFont === undefined ? REFERENCE_SOURCES.appleEmoji.path : options.appleEmojiFont;
    this.fontPath = font && existsSync(font) ? font : undefined;
    this.timeoutMs = options.timeoutMs ?? 20_000;
  }

  /** The best emoji and icons for a request, highest score first within each kind. */
  async search(text: string, limitPerKind = 6): Promise<ReferenceCandidate[]> {
    const terms = referenceQueryTerms(text);
    if (terms.length === 0) return [];
    const [emoji, icons] = await Promise.all([this.loadEmojiIndex(), this.loadIconIndex()]);
    const rank = <T>(items: T[], toCandidate: (t: T) => ReferenceCandidate) =>
      items
        .map(toCandidate)
        .filter((c) => c.score > 0)
        .sort((a, b) => b.score - a.score || a.label.length - b.label.length)
        .slice(0, limitPerKind);
    return [
      ...rank(emoji, (e) => ({ id: `emoji:${e.hexcode}`, kind: "emoji", label: e.label, tags: e.tags ?? [], score: scoreCandidate(terms, e.label, e.tags ?? []) })),
      ...rank(icons, (i) => ({ id: `icon:${i.name}`, kind: "icon", label: i.name.replace(/_/g, " "), tags: i.tags, score: scoreCandidate(terms, i.name.replace(/_/g, " "), i.tags) })),
    ];
  }

  /** Render a candidate (or an id from {@link search}) as a transparent PNG of about `size` px. */
  async load(id: string, size = 512): Promise<ReferenceAsset> {
    const [kind, key] = splitId(id);
    if (kind === "icon") {
      const index = await this.loadIconIndex();
      const entry = index.find((i) => i.name === key);
      if (!entry) throw new ComfyAgentError("GRAPH_EDIT", `no Material Symbol named "${key}"`);
      let svg: string | undefined;
      let url = "";
      for (const filled of [true, false]) {
        url = REFERENCE_SOURCES.materialSvg.url(key, filled);
        svg = await this.cachedText(url, path.join("material-symbols", `${key}${filled ? "-fill" : ""}.svg`)).catch(() => undefined);
        if (svg) break;
      }
      if (!svg) throw new ComfyAgentError("GRAPH_EDIT", `could not fetch Material Symbol "${key}"`);
      return { id, kind, label: key.replace(/_/g, " "), source: REFERENCE_SOURCES.materialSvg.label, url, png: encodePng(rasterizeSvg(svg, size, { padding: 0.1 })) };
    }
    const index = await this.loadEmojiIndex();
    const entry = index.find((e) => e.hexcode === key);
    if (!entry) throw new ComfyAgentError("GRAPH_EDIT", `no emoji with code ${key}`);
    const codepoints = key.split("-").map((h) => parseInt(h, 16));
    for (const renderer of this.renderers) {
      const got = await this.renderEmoji(renderer, entry, codepoints).catch(() => undefined);
      if (got) {
        return { id, kind, label: entry.label, source: got.source, ...(got.url ? { url: got.url } : {}), png: encodePng(fitSquare(decodePng(got.png), size)) };
      }
    }
    throw new ComfyAgentError("GRAPH_EDIT", `no emoji image source has ${entry.label} (${key})`);
  }

  private async renderEmoji(renderer: EmojiRenderer, entry: EmojiEntry, codepoints: number[]): Promise<{ png: Buffer; source: string; url?: string } | undefined> {
    if (renderer === "apple") {
      const font = await this.appleFont();
      const png = font?.png(codepoints);
      return png ? { png, source: REFERENCE_SOURCES.appleEmoji.label } : undefined;
    }
    if (renderer === "fluent") {
      const folder = entry.label.charAt(0).toUpperCase() + entry.label.slice(1);
      const stem = entry.label.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
      for (const rel of [`3D/${stem}_3d.png`, `Default/3D/${stem}_3d_default.png`]) {
        const url = REFERENCE_SOURCES.fluentEmoji.url(folder, rel);
        const png = await this.cachedBytes(url, path.join("fluent-emoji", `${entry.hexcode}.png`)).catch(() => undefined);
        if (png) return { png, source: REFERENCE_SOURCES.fluentEmoji.label, url };
      }
      return undefined;
    }
    const file = `emoji_u${codepoints.filter((c) => c !== 0xfe0f).map((c) => c.toString(16).padStart(4, "0")).join("_")}.png`;
    const url = REFERENCE_SOURCES.notoEmoji.url(file);
    const png = await this.cachedBytes(url, path.join("noto-emoji", file));
    return { png, source: REFERENCE_SOURCES.notoEmoji.label, url };
  }

  private async appleFont(): Promise<SbixFont | null> {
    if (this.font !== undefined) return this.font;
    this.font = this.fontPath ? openSbixFont(await readFile(this.fontPath)) : null;
    return this.font;
  }

  private loadEmojiIndex(): Promise<EmojiEntry[]> {
    this.emojiIndex ??= this.cachedText(REFERENCE_SOURCES.emojiIndex.url, path.join("index", "emojibase-17.0.0-en-compact.json")).then((text) => {
      this.indexDigests.emoji = sha256(text);
      // Group 2 is "component" (skin tones, hair): not subjects.
      return (JSON.parse(text) as EmojiEntry[]).filter((e) => e.hexcode && e.label && e.group !== 2);
    });
    return this.emojiIndex;
  }

  private loadIconIndex(): Promise<IconEntry[]> {
    this.iconIndex ??= this.cachedText(REFERENCE_SOURCES.materialIndex.url, path.join("index", "material-symbols-metadata.json")).then((text) => {
      this.indexDigests.icons = sha256(text);
      const body = JSON.parse(text.replace(/^\)\]\}'\s*/, "")) as { icons: (IconEntry & { unsupported_families?: string[] })[] };
      const seen = new Set<string>();
      const out: IconEntry[] = [];
      for (const i of body.icons) {
        if (i.unsupported_families?.includes("Material Symbols Outlined") || seen.has(i.name)) continue;
        seen.add(i.name);
        out.push({ name: i.name, tags: i.tags ?? [], ...(i.categories ? { categories: i.categories } : {}) });
      }
      return out;
    });
    return this.iconIndex;
  }

  private async cachedBytes(url: string, rel: string): Promise<Buffer> {
    const file = path.join(this.cacheDir, rel);
    if (existsSync(file)) return readFile(file);
    // A pinned URL that answered 404 stays missing; do not ask again.
    const missing = `${file}.${createHash("sha256").update(url).digest("hex").slice(0, 12)}.missing`;
    if (existsSync(missing)) throw new ComfyAgentError("GRAPH_EDIT", `reference not found (cached): ${url}`);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(url, { signal: controller.signal });
      if (res.status === 404) {
        await mkdir(path.dirname(missing), { recursive: true });
        await writeFile(missing, url);
      }
      if (!res.ok) throw new ComfyAgentError("GRAPH_EDIT", `reference fetch ${res.status}: ${url}`);
      const bytes = Buffer.from(await res.arrayBuffer());
      await mkdir(path.dirname(file), { recursive: true });
      await writeFile(file, bytes);
      return bytes;
    } finally {
      clearTimeout(timer);
    }
  }

  private async cachedText(url: string, rel: string): Promise<string> {
    return (await this.cachedBytes(url, rel)).toString("utf8");
  }
}

function splitId(id: string): ["emoji" | "icon", string] {
  const m = /^(emoji|icon):(.+)$/.exec(id);
  if (!m) throw new ComfyAgentError("GRAPH_EDIT", `reference id must be emoji:<hexcode> or icon:<name>, not "${id}"`);
  return [m[1] as "emoji" | "icon", m[2]!];
}

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

/** Centre an image in a transparent square of `size` px, scaled to fit. */
function fitSquare(img: RgbaImage, size: number): RgbaImage {
  const scale = size / Math.max(img.width, img.height);
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const scaled = resize(img, w, h);
  const out = createImage(size, size, { r: 0, g: 0, b: 0, a: 0 });
  const ox = Math.floor((size - w) / 2);
  const oy = Math.floor((size - h) / 2);
  for (let y = 0; y < h; y += 1) out.data.set(scaled.data.subarray(y * w * 4, (y + 1) * w * 4), ((oy + y) * size + ox) * 4);
  return out;
}
