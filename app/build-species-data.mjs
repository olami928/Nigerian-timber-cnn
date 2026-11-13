import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Generates the bundled offline species database from authoritative project sources.
 * Sources (never fetched at runtime): configs/species.csv, configs/species_enduse.json, data/manifest.csv
 * Output: app/public/species.json (copied into Android assets by build/sync)
 *
 * Regenerate whenever source files change:
 *   npm --prefix app run build
 * (or: node app/build-species-data.mjs)
 */
const appDir = dirname(fileURLToPath(import.meta.url));
const projectDir = resolve(appDir, "..");
const csv = await readFile(resolve(projectDir, "configs/species.csv"), "utf8");
const lookup = JSON.parse(
  await readFile(resolve(projectDir, "configs/species_enduse.json"), "utf8"),
);
const lookupBySlug = new Map(lookup.species.map((species) => [species.slug, species]));
const rows = csv.trim().split(/\r?\n/).slice(1);
const manifestText = await readFile(resolve(projectDir, "data/manifest.csv"), "utf8");
const manifestRows = manifestText.trim().split(/\r?\n/).map(parseCsvRow);
const manifestHeader = manifestRows.shift();
const speciesIndex = manifestHeader.indexOf("species");
const specimenIndex = manifestHeader.indexOf("specimen_code");
if (speciesIndex < 0 || specimenIndex < 0) {
  throw new Error("The manifest must include species and specimen_code columns.");
}
const specimenSets = new Map();
for (const row of manifestRows) {
  const species = row[speciesIndex];
  const specimen = row[specimenIndex];
  if (!species || !specimen) throw new Error("The manifest contains an empty species/specimen.");
  if (!specimenSets.has(species)) specimenSets.set(species, new Set());
  specimenSets.get(species).add(specimen);
}

const species = rows.map((row) => {
  const [id, slug, localName, botanical, woodType] = parseCsvRow(row);
  const details = lookupBySlug.get(slug);
  if (!details) {
    throw new Error(`Missing species_enduse.json entry for: ${slug}`);
  }
  if (details.botanical !== botanical) {
    throw new Error(`Species lookup botanical does not match configs/species.csv: ${slug}`);
  }
  if ((details.local_name ?? "") !== localName || (details.wood_type ?? "") !== woodType) {
    throw new Error(`Species lookup local_name/wood_type does not match configs/species.csv: ${slug}`);
  }
  const documentedUses = Object.fromEntries(
    Object.entries(details.uses ?? {}).filter(([, status]) =>
      ["primary", "secondary"].includes(status),
    ),
  );
  const sources = Array.isArray(details.sources)
    ? details.sources.filter((value) => typeof value === "string" && value.trim().length > 0)
    : [];
  return {
    id: Number(id),
    slug,
    local_name: localName,
    botanical,
    english_name: details.english_name ?? null,
    wood_type: woodType,
    documented_uses: documentedUses,
    sources,
    training_specimens: specimenSets.get(slug)?.size ?? 0,
    provenance: {
      status: details.status ?? null,
      notes: details.notes ?? null,
    },
  };
});

if (
  species.length !== 29
  || lookupBySlug.size !== species.length
  || species.some((item) => item.training_specimens < 1)
  || species.some((item, index) => item.id !== index + 1)
) {
  throw new Error("Expected matching species lists of exactly 29 entries with canonical IDs.");
}

const database = {
  meta: {
    generated_from: [
      "configs/species.csv",
      "configs/species_enduse.json",
      "data/manifest.csv",
    ],
    species_count: species.length,
    purpose:
      "Bundled offline species reference. Documented uses are literature/database records, not model predictions.",
    rules: Array.isArray(lookup.meta?.rules) ? lookup.meta.rules : [],
    status_values: lookup.meta?.status_values ?? {},
  },
  species,
};

await writeFile(
  resolve(appDir, "public/species.json"),
  `${JSON.stringify(database, null, 2)}\n`,
);
console.log(`Generated offline species database for ${species.length} species.`);

function parseCsvRow(line) {
  const fields = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      fields.push(field);
      field = "";
    } else {
      field += character;
    }
  }
  if (quoted) throw new Error("The manifest contains an unterminated quoted field.");
  fields.push(field);
  return fields;
}
