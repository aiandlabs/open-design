import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { alphaStats, decodePng } from "../src/image.js";
import { REFERENCE_SOURCES, ReferenceLibrary, referenceQueryTerms, scoreCandidate } from "../src/references.js";
import { solidPng } from "./fixtures/fake-comfy.js";

const dirs: string[] = [];
afterEach(async () => {
  await Promise.all(dirs.splice(0).map((d) => rm(d, { recursive: true, force: true })));
});

const EMOJI = [
  { hexcode: "2615", label: "hot beverage", tags: ["beverage", "coffee", "drink", "hot", "steaming", "tea"], group: 4 },
  { hexcode: "1F3EE", label: "red paper lantern", tags: ["bar", "lantern", "light", "red"], group: 7 },
  { hexcode: "1F680", label: "rocket", tags: ["space"], group: 5 },
  { hexcode: "1F3FB", label: "light skin tone", tags: ["skin", "tone"], group: 2 },
];
const ICONS = {
  icons: [
    { name: "coffee", tags: ["beverage", "coffee", "cup", "drink", "mug", "steam"], unsupported_families: ["Material Icons"] },
    { name: "coffee_maker", tags: ["appliance", "coffee"], unsupported_families: [] },
    { name: "coffee", tags: ["cup"], unsupported_families: ["Material Symbols Outlined"] },
    { name: "rocket_launch", tags: ["rocket", "launch", "space"], unsupported_families: [] },
  ],
};
const SQUARE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M160-800H800V-160H160Z"/></svg>';

function fakeFetch(opts: { fluent?: boolean; noto?: boolean } = {}) {
  const calls: string[] = [];
  const impl = (async (input: string | URL | Request) => {
    const url = String(input);
    calls.push(url);
    if (url === REFERENCE_SOURCES.emojiIndex.url) return new Response(JSON.stringify(EMOJI));
    if (url === REFERENCE_SOURCES.materialIndex.url) return new Response(`)]}'\n${JSON.stringify(ICONS)}`);
    if (url.includes("@material-symbols/svg-400")) return url.includes("-fill.svg") ? new Response(SQUARE_SVG) : new Response("", { status: 404 });
    if (url.includes("fluentui-emoji")) return opts.fluent ? new Response(solidPng(256, 256, [120, 70, 30])) : new Response("", { status: 404 });
    if (url.includes("noto-emoji")) return opts.noto === false ? new Response("", { status: 404 }) : new Response(solidPng(512, 512, [90, 50, 20]));
    return new Response("", { status: 404 });
  }) as typeof fetch;
  return { impl, calls };
}

async function library(fetchOpts: Parameters<typeof fakeFetch>[0] = {}) {
  const cacheDir = await mkdtemp(path.join(os.tmpdir(), "refs-"));
  dirs.push(cacheDir);
  const f = fakeFetch(fetchOpts);
  return { lib: new ReferenceLibrary({ cacheDir, fetch: f.impl, appleEmojiFont: false }), calls: f.calls, cacheDir };
}

describe("reference search", () => {
  it("keeps the request's subject words and drops style, colour and layout words", () => {
    expect(referenceQueryTerms("A coffee cup icon with steam, flat style, brown and cream.")).toEqual(["coffee", "cup", "steam"]);
    expect(referenceQueryTerms("A small rocket icon pointing up and to the right")).toEqual(["rocket"]);
    expect(referenceQueryTerms("Two paper lanterns")).toEqual(["two", "paper", "lantern"]);
  });

  it("prefers names over tags, and shorter names over longer ones", () => {
    const terms = ["coffee"];
    expect(scoreCandidate(terms, "coffee", [])).toBeGreaterThan(scoreCandidate(terms, "coffee maker", []));
    expect(scoreCandidate(terms, "coffee maker", [])).toBeGreaterThan(scoreCandidate(terms, "hot beverage", ["coffee"]));
    expect(scoreCandidate(terms, "rocket", ["space"])).toBe(0);
  });

  it("returns ranked emoji and Material Symbols for a request, skipping skin-tone components and Symbols-less icons", async () => {
    const { lib } = await library();
    const found = await lib.search("A coffee cup icon with steam, flat style");
    expect(found.filter((c) => c.kind === "emoji").map((c) => c.id)).toEqual(["emoji:2615"]);
    expect(found.filter((c) => c.kind === "icon").map((c) => c.id)).toEqual(["icon:coffee", "icon:coffee_maker"]);
    expect((await lib.search("skin tone")).some((c) => c.id === "emoji:1F3FB")).toBe(false);
    expect(lib.indexDigests.emoji).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe("reference rendering", () => {
  it("rasterises the filled Material Symbol to a transparent PNG and records its source", async () => {
    const { lib } = await library();
    const ref = await lib.load("icon:coffee", 256);
    expect(ref.source).toBe(REFERENCE_SOURCES.materialSvg.label);
    expect(ref.url).toContain("coffee-fill.svg");
    const img = decodePng(ref.png);
    expect([img.width, img.height]).toEqual([256, 256]);
    expect(alphaStats(img).borderTransparent).toBe(1);
    expect(img.data[(128 * 256 + 128) * 4 + 3]).toBe(255);
  });

  it("tries emoji sources in order and caches what it fetched", async () => {
    const { lib, calls } = await library({ fluent: false });
    const ref = await lib.load("emoji:2615", 128);
    expect(ref.source).toBe(REFERENCE_SOURCES.notoEmoji.label);
    expect(ref.url).toContain("emoji_u2615.png");
    expect(calls.filter((c) => c.includes("fluentui-emoji"))).toHaveLength(2); // 3D, then Default/3D
    const fetched = calls.length;
    await lib.load("emoji:2615", 128);
    expect(calls.length).toBe(fetched);
  });

  it("uses Fluent Emoji 3D before Noto when it has the emoji", async () => {
    const { lib, calls } = await library({ fluent: true });
    const ref = await lib.load("emoji:1F3EE", 128);
    expect(ref.source).toBe(REFERENCE_SOURCES.fluentEmoji.label);
    expect(calls.some((c) => c.includes("Red%20paper%20lantern/3D/red_paper_lantern_3d.png"))).toBe(true);
  });

  it("fails clearly for unknown ids", async () => {
    const { lib } = await library();
    await expect(lib.load("emoji:FFFF")).rejects.toThrow(/no emoji/);
    await expect(lib.load("nonsense")).rejects.toThrow(/emoji:<hexcode> or icon:<name>/);
  });
});
