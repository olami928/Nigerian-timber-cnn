import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

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
  const [id, slug, localName, botanical, woodType] = row.split(",");
  const details = lookupBySlug.get(slug);
  if (!details || details.botanical !== botanical) {
    throw new Error(`Species lookup does not match configs/species.csv: ${slug}`);
  }
  const uses = Object.fromEntries(
    Object.entries(details.uses ?? {}).filter(([, status]) =>
      ["primary", "secondary"].includes(status),
    ),
  );
  return {
    id: Number(id),
    slug,
    local_name: localName,
    botanical,
    english_name: details.english_name || null,
    wood_type: woodType,
    uses,
    training_specimens: specimenSets.get(slug)?.size ?? 0,
  };
});

if (
  species.length !== 29
  || lookupBySlug.size !== species.length
  || species.some((item) => item.training_specimens < 1)
) {
  throw new Error("Expected matching species lists of exactly 29 entries.");
}

await writeFile(
  resolve(appDir, "public/species.json"),
  `${JSON.stringify(species, null, 2)}\n`,
);
console.log(`Generated public species data for ${species.length} species.`);

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
