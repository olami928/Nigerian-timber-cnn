import { access, copyFile, mkdir, readFile, readdir, unlink } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appDir = dirname(fileURLToPath(import.meta.url));
const publicDir = resolve(appDir, "public");
const modelDir = resolve(appDir, "model");
const modelFiles = [
  ["model.onnx", "model.onnx"],
  ["class_names.json", "class_names.json"],
  ["preprocess_config.json", "preprocess_config.json"],
];

await import("./build-species-data.mjs");

for (const [source, destination] of modelFiles) {
  await copyFile(resolve(modelDir, source), resolve(publicDir, destination));
}

const [config, classNames, speciesDb] = await Promise.all([
  readFile(resolve(publicDir, "preprocess_config.json"), "utf8").then(JSON.parse),
  readFile(resolve(publicDir, "class_names.json"), "utf8").then(JSON.parse),
  readFile(resolve(publicDir, "species.json"), "utf8").then(JSON.parse),
]);
const species = Array.isArray(speciesDb?.species) ? speciesDb.species : speciesDb;
if (
  config.min_sharpness !== 100
  || !Array.isArray(classNames)
  || classNames.length !== 29
  || !Array.isArray(species)
  || species.length !== 29
  || classNames.some((name, index) => name !== species[index]?.slug)
) {
  throw new Error("The model classes, species list, or approved sharpness cutoff do not match.");
}

const wasmSourceDir = resolve(appDir, "node_modules/onnxruntime-web/dist");
const wasmFiles = [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
];
for (const wasmFile of wasmFiles) {
  const source = resolve(wasmSourceDir, wasmFile);
  try {
    await access(source);
  } catch (error) {
    throw new Error(`Required ONNX Runtime Web asset is missing: ${source}`, { cause: error });
  }
}
const wasmOutputDir = resolve(publicDir, "ort");
await mkdir(wasmOutputDir, { recursive: true });
for (const name of await readdir(wasmOutputDir)) {
  if (!wasmFiles.includes(name)) {
    await unlink(resolve(wasmOutputDir, name));
  }
}
for (const wasmFile of wasmFiles) {
  await copyFile(resolve(wasmSourceDir, wasmFile), resolve(wasmOutputDir, wasmFile));
}

const { build } = await import("esbuild");
await build({
  entryPoints: [resolve(publicDir, "app.js")],
  bundle: true,
  outfile: resolve(publicDir, "app.bundle.js"),
  platform: "browser",
  format: "iife",
  target: "es2022",
  conditions: ["onnxruntime-web-use-extern-wasm"],
});
console.log(`Built browser model app and copied ${wasmFiles.length} WASM runtime files.`);
