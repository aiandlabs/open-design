import { build } from "esbuild";

const shared = {
  bundle: true,
  format: "esm",
  packages: "external",
  platform: "node",
  target: "node24",
};

await build({ ...shared, entryPoints: ["./src/index.ts"], outfile: "./dist/index.mjs" });
await build({
  ...shared,
  banner: { js: "#!/usr/bin/env node" },
  entryPoints: ["./src/bin.ts"],
  outfile: "./dist/bin.mjs",
});
