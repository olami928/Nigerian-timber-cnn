import * as ort from "onnxruntime-web/wasm";
import { calculateSharpness, makeTensorData } from "./inference-core.mjs";

const screens = [...document.querySelectorAll(".screen")];
const navLinks = [...document.querySelectorAll(".nav-link")];
const speciesList = document.querySelector("#species-list");
const speciesSearch = document.querySelector("#species-search");
const speciesCount = document.querySelector("#species-count");
const speciesEmpty = document.querySelector("#species-empty");
const speciesError = document.querySelector("#species-error");
const uploadDrop = document.querySelector("#upload-drop");
const fileInput = document.querySelector("#wood-photo");
const previewWrap = document.querySelector("#preview-wrap");
const preview = document.querySelector("#photo-preview");
const photoMessage = document.querySelector("#photo-message");
const identifyButton = document.querySelector("#identify-button");
const identifyStatus = document.querySelector("#identify-status");
const result = document.querySelector("#result");

const MAX_FILE_BYTES = 8 * 1024 * 1024;
const IMAGE_TYPES = new Set(["image/jpeg", "image/png"]);
const GRID_ROWS = 6;
const GRID_COLS = 8;
const PATCH_COUNT = GRID_ROWS * GRID_COLS;
const PATCH_SIZE = 112;
const speciesByName = new Map();
let speciesData = [];
let speciesPromise = Promise.resolve(null);
let selectedImage;
let selectedImageData;
let selectedSharpness;
let previewUrl;
let modelPromise;
let busy = false;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "/ort/";

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function showScreen(name) {
  screens.forEach((screen) => {
    const active = screen.id === `screen-${name}`;
    screen.classList.toggle("active", active);
    screen.hidden = !active;
  });
  navLinks.forEach((link) => {
    const active = link.dataset.screen === name;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  window.scrollTo({ top: 0, behavior: reduceMotion.matches ? "auto" : "smooth" });
}

navLinks.forEach((link) =>
  link.addEventListener("click", () => showScreen(link.dataset.screen)),
);
document.querySelectorAll("[data-go]").forEach((button) =>
  button.addEventListener("click", () => showScreen(button.dataset.go)),
);

function renderSpecies(query = "") {
  const normalized = query.trim().toLocaleLowerCase();
  const matches = speciesData.filter((species) =>
    `${species.local_name} ${species.botanical}`.toLocaleLowerCase().includes(normalized),
  );
  speciesList.replaceChildren();
  matches.forEach((species) => {
    const card = document.createElement("article");
    card.className = "species-card";
    const id = document.createElement("span");
    id.className = "species-id";
    id.textContent = String(species.id).padStart(2, "0");
    const name = document.createElement("h2");
    name.textContent = species.local_name;
    const botanical = document.createElement("p");
    botanical.className = "scientific";
    botanical.textContent = species.botanical;
    const type = document.createElement("span");
    type.className = "badge";
    type.textContent = species.wood_type;
    card.append(id, name, botanical, type);
    speciesList.append(card);
  });
  speciesEmpty.hidden = matches.length !== 0;
  speciesCount.textContent = `${matches.length} of ${speciesData.length} species`;
}

speciesPromise = fetch("/species.json")
  .then((response) => {
    if (!response.ok) throw new Error(`Species data request failed: ${response.status}`);
    return response.json();
  })
  .then((species) => {
    if (!Array.isArray(species) || species.length !== 29) {
      throw new Error("Species data did not contain the expected 29 entries.");
    }
    speciesData = species;
    species.forEach((item) => speciesByName.set(item.slug, item));
    renderSpecies();
    return species;
  })
  .catch((error) => {
    console.error(error);
    speciesCount.hidden = true;
    speciesError.hidden = false;
    return null;
  });

speciesSearch.addEventListener("input", (event) => renderSpecies(event.target.value));

function clearSelection(message) {
  selectedImage?.close();
  selectedImage = undefined;
  selectedImageData = undefined;
  selectedSharpness = undefined;
  identifyButton.disabled = true;
  previewWrap.hidden = true;
  photoMessage.textContent = message;
  result.replaceChildren();
  identifyStatus.textContent = "";
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = undefined;
  }
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  fileInput.value = "";
  clearSelection("");
  if (!IMAGE_TYPES.has(file.type)) {
    photoMessage.textContent = "Choose a JPEG or PNG image.";
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    photoMessage.textContent = "This image is larger than 8 MB. Choose a smaller file.";
    return;
  }

  try {
    const bitmap = await createImageBitmap(file, {
      colorSpaceConversion: "none",
      imageOrientation: "from-image",
    });
    if (bitmap.width < GRID_COLS || bitmap.height < GRID_ROWS) {
      bitmap.close();
      photoMessage.textContent = "This image is too small to analyse 48 sections. Choose a larger image.";
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Your browser could not prepare the selected image.");
    context.drawImage(bitmap, 0, 0);
    selectedImageData = context.getImageData(0, 0, canvas.width, canvas.height);
    selectedSharpness = calculateSharpness(selectedImageData);
    selectedImage = bitmap;
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    previewWrap.hidden = false;
    identifyButton.disabled = false;
    photoMessage.textContent = selectedSharpness < 100
      ? `This image may be too blurry (sharpness ${selectedSharpness.toFixed(1)}; minimum 100). You can retake it or continue to see the quality check.`
      : `Image ready (${bitmap.width} × ${bitmap.height}). Images stay in this browser; sharpness ${selectedSharpness.toFixed(1)}.`;
  } catch (error) {
    console.error(error);
    photoMessage.textContent = error instanceof Error
      ? `The image could not be opened: ${error.message}`
      : "The image could not be opened. Try a JPEG or PNG file.";
  }
});

if (uploadDrop) {
  uploadDrop.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadDrop.classList.add("is-dragging");
  });
  uploadDrop.addEventListener("dragleave", (event) => {
    if (!uploadDrop.contains(event.relatedTarget)) {
      uploadDrop.classList.remove("is-dragging");
    }
  });
  uploadDrop.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadDrop.classList.remove("is-dragging");
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    fileInput.dispatchEvent(new Event("change"));
  });
}

async function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const [configResponse, classesResponse, modelResponse] = await Promise.all([
        fetch("/preprocess_config.json"),
        fetch("/class_names.json"),
        fetch("/model.onnx"),
      ]);
      for (const [label, response] of [
        ["Model configuration", configResponse],
        ["Class names", classesResponse],
        ["Model file", modelResponse],
      ]) {
        if (!response.ok) throw new Error(`${label} could not be loaded (${response.status}).`);
      }
      const [config, classNames, modelBuffer] = await Promise.all([
        configResponse.json(),
        classesResponse.json(),
        modelResponse.arrayBuffer(),
      ]);
      if (
        config.architecture !== "shufflenet_v2"
        || config.grid_rows !== GRID_ROWS
        || config.grid_cols !== GRID_COLS
        || config.num_patches !== PATCH_COUNT
        || config.patch_size[0] !== PATCH_SIZE
        || config.patch_size[1] !== PATCH_SIZE
        || config.min_sharpness !== 171.4
        || !Array.isArray(classNames)
        || classNames.length !== 29
      ) {
        throw new Error("The deployed model configuration does not match the approved setup.");
      }
      const session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      if (session.inputNames.length !== 1 || session.outputNames.length !== 1) {
        throw new Error("The deployed model has an unexpected input/output signature.");
      }
      return { session, config, classNames };
    })().catch((error) => {
      modelPromise = undefined;
      throw error;
    });
  }
  return modelPromise;
}

function makeTensor(imageData, config) {
  const values = makeTensorData(imageData, config.normalization, GRID_ROWS, GRID_COLS, PATCH_SIZE);
  return new ort.Tensor("float32", values, [PATCH_COUNT, 3, PATCH_SIZE, PATCH_SIZE]);
}

function roundOneDecimal(value) {
  const scaled = value * 10;
  const lower = Math.floor(scaled);
  const fraction = scaled - lower;
  const isTie = Math.abs(fraction - 0.5) <= Number.EPSILON * Math.max(1, Math.abs(scaled));
  if (isTie) return (lower % 2 === 0 ? lower : lower + 1) / 10;
  return Math.round(scaled) / 10;
}

function aggregate(logits, classNames, config) {
  const classCount = classNames.length;
  const counts = new Uint32Array(classCount);
  const meanProbabilities = new Float64Array(classCount);
  for (let patch = 0; patch < PATCH_COUNT; patch += 1) {
    const offset = patch * classCount;
    let maximum = -Infinity;
    let winner = 0;
    for (let index = 0; index < classCount; index += 1) {
      if (logits[offset + index] > maximum) {
        maximum = logits[offset + index];
        winner = index;
      }
    }
    counts[winner] += 1;
    let denominator = 0;
    for (let index = 0; index < classCount; index += 1) {
      denominator += Math.exp(logits[offset + index] - maximum);
    }
    for (let index = 0; index < classCount; index += 1) {
      meanProbabilities[index] += Math.exp(logits[offset + index] - maximum) / denominator;
    }
  }
  for (let index = 0; index < classCount; index += 1) {
    meanProbabilities[index] /= PATCH_COUNT;
  }
  const order = Array.from({ length: classCount }, (_, index) => index).sort(
    (left, right) =>
      (counts[right] + 1e-3 * meanProbabilities[right])
      - (counts[left] + 1e-3 * meanProbabilities[left])
      || right - left,
  );
  const winner = order[0];
  const species = speciesByName.get(classNames[winner]);
  if (!species) throw new Error(`No documented species data for ${classNames[winner]}.`);
  const top3 = order.slice(0, 3).map((index) => {
    if (!speciesByName.has(classNames[index])) {
      throw new Error(`No documented species data for ${classNames[index]}.`);
    }
    return {
      species: classNames[index],
      vote_pct: roundOneDecimal((100 * counts[index]) / PATCH_COUNT),
      prob_pct: roundOneDecimal(100 * meanProbabilities[index]),
    };
  });
  return {
    status: "ok",
    species: classNames[winner],
    confidence_pct: roundOneDecimal((100 * counts[winner]) / PATCH_COUNT),
    patch_agreement: `${counts[winner]}/${PATCH_COUNT}`,
    mean_probability_pct: roundOneDecimal(100 * meanProbabilities[winner]),
    top3,
    limited_reference_data: species.training_specimens < 3,
    model_version: config.model_version,
    local_name: species.local_name,
    botanical: species.botanical,
    english_name: species.english_name,
    wood_type: species.wood_type,
    documented_uses: Object.fromEntries(
      Object.entries(species.uses ?? {}).filter(
        ([, status]) => status === "primary" || status === "secondary",
      ),
    ),
  };
}

function element(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function revealResult(card) {
  result.replaceChildren(card);
  const heading = card.querySelector("h2");
  if (heading) {
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
  }
  card.scrollIntoView({ block: "start", behavior: reduceMotion.matches ? "auto" : "smooth" });
}

function renderRetake(reason, top3) {
  const card = element("section", "result-card retake-card");
  card.setAttribute("aria-labelledby", "result-title");
  const heading = element("h2", "", "Please retake the photo");
  heading.id = "result-title";
  card.append(heading);
  card.append(element(
    "p",
    "",
    reason === "blurry"
      ? "The cut surface looks too blurry. Hold steady, refocus, and try again."
      : "The model is not certain enough to name a species. Try a flatter, cleaner cut surface.",
  ));
  if (top3?.length) {
    card.append(element("h3", "", "Possible matches — not certain"));
    const list = element("ol", "top-list");
    top3.forEach((item) => {
      const species = speciesByName.get(item.species);
      list.append(element(
        "li",
        "",
        `${species?.local_name ?? item.species}: ${item.vote_pct}% patch agreement`,
      ));
    });
    card.append(list);
  }
  card.append(makeRetakeButton());
  revealResult(card);
}

function makeRetakeButton() {
  const button = element("button", "button button-secondary", "Choose another photo");
  button.type = "button";
  button.addEventListener("click", () => fileInput.click());
  return button;
}

function renderPrediction(prediction) {
  const card = element("section", "result-card");
  card.setAttribute("aria-labelledby", "result-title");
  const species = speciesByName.get(prediction.species);
  if (!species) throw new Error(`Species details are missing for ${prediction.species}.`);
  const heading = element("h2", "", species.local_name);
  heading.id = "result-title";
  card.append(heading);
  const scientific = element("p", "scientific", species.botanical);
  card.append(scientific);
  if (species.english_name) card.append(element("p", "english-name", species.english_name));
  card.append(element("span", "badge", "Hardwood"));

  const confidence = element("div", "confidence-block");
  const label = element("p", "confidence-label");
  label.append(
    element("strong", "", `Patch agreement: ${prediction.confidence_pct}%`),
    element("span", "", `${prediction.patch_agreement} image sections agree`),
  );
  const progress = element("progress", "confidence-bar");
  progress.max = 100;
  progress.value = prediction.confidence_pct;
  progress.setAttribute("aria-label", "Patch agreement percentage");
  confidence.append(label, progress);
  confidence.append(element(
    "p",
    "supporting-text",
    `Mean winning-class probability (secondary): ${prediction.mean_probability_pct}%. Patch agreement is not a guarantee of correctness.`,
  ));
  card.append(confidence);

  const top = element("div", "top-matches");
  top.append(element("h3", "", "Top matches"));
  const list = element("ol", "top-list");
  prediction.top3.forEach((item) => {
    const entry = element("li");
    const name = speciesByName.get(item.species)?.local_name ?? item.species;
    entry.append(
      element("span", "", name),
      element("span", "top-match-values", `${item.vote_pct}% vote · ${item.prob_pct}% probability`),
    );
    list.append(entry);
  });
  top.append(list);
  card.append(top);

  if (prediction.limited_reference_data) {
    card.append(element(
      "p",
      "notice",
      "This species was trained on fewer than 3 specimens. Treat the result with extra caution.",
    ));
  }

  const uses = Object.entries(species.uses ?? {}).filter(
    ([, status]) => status === "primary" || status === "secondary",
  );
  if (uses.length) {
    const useSection = element("section", "uses-section");
    useSection.append(element("h3", "", "Documented uses"));
    const legend = element("p", "use-legend", "Solid: primary · Outlined: secondary");
    const chips = element("div", "use-chips");
    uses.forEach(([use, status]) => {
      chips.append(element(
        "span",
        `use-chip ${status}`,
        use.replaceAll("_", " "),
      ));
    });
    useSection.append(chips, legend, element(
      "p",
      "supporting-text",
      "Documented uses from the literature, not predicted from your photo.",
    ));
    card.append(useSection);
  }
  card.append(element("p", "scope-note", "Covers 29 species from Ibadan wood markets only."));
  card.append(makeRetakeButton());
  revealResult(card);
}

function showError(message) {
  const card = element("section", "result-card error-card");
  const heading = element("h2", "", "Could not identify this image");
  heading.id = "result-title";
  card.setAttribute("aria-labelledby", "result-title");
  card.append(heading);
  card.append(element("p", "", message));
  const retry = element("button", "button button-secondary", "Try again");
  retry.type = "button";
  retry.addEventListener("click", () => identifyButton.click());
  card.append(retry);
  revealResult(card);
}

identifyButton.addEventListener("click", async () => {
  if (!selectedImageData || selectedSharpness === undefined || busy) return;
  result.replaceChildren();
  if (selectedSharpness < 100) {
    identifyStatus.textContent = "";
    renderRetake("blurry");
    return;
  }

  busy = true;
  identifyButton.disabled = true;
  fileInput.disabled = true;
  identifyButton.textContent = "Analysing…";
  identifyStatus.textContent = "Analysing 48 sections of your image. Your image stays in this browser.";
  const spinner = element("span", "spinner");
  spinner.setAttribute("aria-hidden", "true");
  identifyStatus.prepend(spinner);
  try {
    const { session, config, classNames } = await loadModel();
    if (!await speciesPromise) {
      throw new Error("Species information could not be loaded. Reload the page and try again.");
    }
    const tensor = makeTensor(selectedImageData, config);
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const logits = outputs[session.outputNames[0]]?.data;
    if (!logits || logits.length !== PATCH_COUNT * classNames.length) {
      throw new Error("The model returned an unexpected set of predictions.");
    }
    const prediction = aggregate(logits, classNames, config);
    renderPrediction(prediction);
    identifyStatus.textContent = `Analysis complete · ${prediction.model_version}`;
  } catch (error) {
    console.error(error);
    showError(error instanceof Error ? error.message : "The model could not process this image.");
    identifyStatus.textContent = "No prediction was produced.";
  } finally {
    busy = false;
    fileInput.disabled = false;
    identifyButton.disabled = !selectedImageData;
    identifyButton.textContent = "Identify species";
  }
});
