import * as ort from "onnxruntime-web/wasm";
import { calculateSharpness, makeTensorData } from "./inference-core.mjs";

/* ==========================================================================
   Configuration & Constants
   ========================================================================== */

const MAX_FILE_BYTES = 8 * 1024 * 1024;
const IMAGE_TYPES = new Set(["image/jpeg", "image/png"]);
const GRID_ROWS = 6;
const GRID_COLS = 8;
const PATCH_COUNT = GRID_ROWS * GRID_COLS;
const PATCH_SIZE = 112;
const MIN_SHARPNESS = 100;
const HISTORY_STORAGE_KEY = "timberid_history";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "./ort/";

const grainColors = [
  "#8d6a2f", "#8a3f2a", "#d9c79b", "#b9832f", "#c7ad74", "#7b5a2c",
  "#6e4726", "#87532b", "#5c3a1e", "#784b28", "#8c562e", "#613b1b",
  "#945b31", "#6a4022", "#754523", "#834d28", "#6f3e1f", "#915629",
  "#7a4422", "#804c26", "#6d3b1e", "#7d4825", "#895029", "#6b3c1d",
  "#774221", "#844a27", "#733e1e", "#864f2a", "#69381c"
];

const cssGrain = (hex) =>
  `repeating-linear-gradient(92deg, rgba(0,0,0,0) 0 7px, rgba(0,0,0,0.12) 8px 9px, rgba(255,255,255,0.06) 10px 15px), ${hex}`;

/* ==========================================================================
   State & Caches
   ========================================================================== */

let speciesList = [];
const speciesBySlug = new Map();
let modelPromise = null;

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
  } catch (err) {
    console.warn("Could not read history from localStorage:", err);
    return [];
  }
}

function saveHistory(items) {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items));
  } catch (err) {
    console.warn("Could not save history to localStorage:", err);
  }
}

function addHistoryEntry(entry) {
  const list = getHistory();
  list.unshift(entry);
  saveHistory(list.slice(0, 50));
}

function clearHistory() {
  try {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
  } catch (err) {
    console.warn("Could not clear history:", err);
  }
}

/* ==========================================================================
   Data & Model Loading
   ========================================================================== */

async function loadSpeciesData() {
  if (speciesList.length > 0) return speciesList;
  const res = await fetch("species.json");
  if (!res.ok) throw new Error(`Failed to load species catalogue (${res.status})`);
  const payload = await res.json();
  const arr = Array.isArray(payload?.species) ? payload.species : payload;
  if (!Array.isArray(arr) || arr.length !== 29) {
    throw new Error(`Catalogue must contain 29 species, found ${arr?.length}`);
  }
  speciesList = arr;
  arr.forEach((item, index) => {
    item.index = index;
    speciesBySlug.set(item.slug, item);
  });
  return speciesList;
}

async function loadModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const [configRes, classesRes, modelRes] = await Promise.all([
        fetch("preprocess_config.json"),
        fetch("class_names.json"),
        fetch("model.onnx"),
      ]);
      if (!configRes.ok || !classesRes.ok || !modelRes.ok) {
        throw new Error("Could not load model files from server.");
      }
      const [config, classNames, modelBuffer] = await Promise.all([
        configRes.json(),
        classesRes.json(),
        modelRes.arrayBuffer(),
      ]);

      if (
        config.architecture !== "shufflenet_v2" ||
        config.grid_rows !== GRID_ROWS ||
        config.grid_cols !== GRID_COLS ||
        config.num_patches !== PATCH_COUNT ||
        config.patch_size[0] !== PATCH_SIZE ||
        config.patch_size[1] !== PATCH_SIZE ||
        config.min_sharpness !== MIN_SHARPNESS ||
        !Array.isArray(classNames) ||
        classNames.length !== 29
      ) {
        throw new Error("Model configuration does not match the approved pipeline.");
      }

      const session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return { session, config, classNames };
    })().catch((err) => {
      modelPromise = null;
      throw err;
    });
  }
  return modelPromise;
}

function formatDocumentedUses(species) {
  const uses = species?.documented_uses ?? species?.uses ?? {};
  const entries = Object.entries(uses).filter(
    ([, status]) => status === "primary" || status === "secondary"
  );
  if (!entries.length) return "None documented in current database";
  return entries.map(([u]) => u.replaceAll("_", " ")).join(", ");
}

/* ==========================================================================
   Inference & Aggregation
   ========================================================================== */

function aggregatePredictions(logits, classNames) {
  const classCount = classNames.length;
  const counts = new Uint32Array(classCount);
  const patchWinners = new Uint32Array(PATCH_COUNT);

  for (let patch = 0; patch < PATCH_COUNT; patch++) {
    const offset = patch * classCount;
    let maxLogit = -Infinity;
    let winningClassIndex = 0;
    for (let c = 0; c < classCount; c++) {
      const val = logits[offset + c];
      if (val > maxLogit) {
        maxLogit = val;
        winningClassIndex = c;
      }
    }
    patchWinners[patch] = winningClassIndex;
    counts[winningClassIndex]++;
  }

  // Sort classes by patch vote count descending
  const order = Array.from({ length: classCount }, (_, i) => i).sort(
    (a, b) => counts[b] - counts[a] || a - b
  );

  const winnerIndex = order[0];
  const winnerSlug = classNames[winnerIndex];
  const winnerSpecies = speciesBySlug.get(winnerSlug);
  if (!winnerSpecies) throw new Error(`Missing species metadata for slug: ${winnerSlug}`);

  const topMatches = order.slice(1, 3).map((idx) => {
    const slug = classNames[idx];
    const sp = speciesBySlug.get(slug);
    return {
      index: idx,
      slug,
      species: sp,
      count: counts[idx],
    };
  });

  return {
    winnerIndex,
    winnerSlug,
    winnerSpecies,
    agreeingCount: counts[winnerIndex],
    patchWinners,
    topMatches,
  };
}

async function prepareImage(file) {
  if (!IMAGE_TYPES.has(file.type)) {
    throw new Error("Use a JPEG or PNG image under 8 MB.");
  }
  if (file.size > MAX_FILE_BYTES) {
    throw new Error("File is larger than 8 MB. Choose a smaller image.");
  }

  const bitmap = await createImageBitmap(file, {
    colorSpaceConversion: "none",
    imageOrientation: "from-image",
  });

  if (bitmap.width < GRID_COLS || bitmap.height < GRID_ROWS) {
    bitmap.close();
    throw new Error("Image resolution is too small to analyse 48 regions.");
  }

  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    bitmap.close();
    throw new Error("Browser could not prepare canvas for analysis.");
  }
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const sharpness = calculateSharpness(imageData);
  bitmap.close();

  return { imageData, sharpness };
}

/* ==========================================================================
   Website Implementation (index.html)
   ========================================================================== */

function initWebsite() {

  // Ensure species-modal is hidden on page load
  const modal = document.getElementById("species-modal");
  if (modal) modal.hidden = true;
  const fig = document.getElementById("fig");
  if (fig) {
    let h = "";
    for (let r = 92; r > 6; r -= 5 + Math.random() * 4) {
      h += `<circle r="${r}" fill="none" stroke="#8A5A33" stroke-width="${0.6 + Math.random() * 0.8}" opacity=".55"/>`;
    }
    for (let a = 0; a < 360; a += 11) {
      h += `<line x1="0" y1="0" x2="${92 * Math.cos((a * Math.PI) / 180)}" y2="${92 * Math.sin((a * Math.PI) / 180)}" stroke="#8A5A33" stroke-width=".4" opacity=".4"/>`;
    }
    let s = 7;
    const rnd = () => (s = (s * 9301 + 49297) % 233280) / 233280;
    for (let i = 0; i < 110; i++) {
      const a = rnd() * 6.283,
        r = 10 + rnd() * 80;
      h += `<ellipse cx="${r * Math.cos(a)}" cy="${r * Math.sin(a)}" rx="${1 + rnd() * 1.8}" ry="${1 + rnd() * 1.8}" fill="#25241F" opacity=".75"/>`;
    }
    fig.innerHTML = h + '<circle r="92" fill="none" stroke="#1F3D2B" stroke-width="2"/>';
  }

  const lib = document.getElementById("lib");
  const q = document.getElementById("q");
  });

  function renderCatalogue() {
    if (!lib) return;
    const term = (q?.value || "").trim().toLowerCase();
    const filtered = speciesList.filter((sp) =>
      `${sp.local_name} ${sp.botanical} ${sp.english_name || ""}`
        .toLowerCase()
        .includes(term)
    );

    if (!filtered.length) {
      lib.innerHTML =
        '<p class="empty" style="grid-column:1/-1">No species match that search. Try a common name or genus.</p>';
      return;
    }

    lib.innerHTML = filtered
      .map(
        (sp) => `
      <article class="card-site" data-slug="${sp.slug}" tabindex="0" role="button" aria-label="${sp.local_name}">
        <div class="sw">
          <img src="species_photos/${sp.slug}.jpg" alt="${sp.local_name}" loading="lazy" onerror="this.style.display='none';this.parentElement.style.background='${cssGrain(grainColors[sp.index % grainColors.length])}'">
        </div>
        <div class="b">
          <h3>${sp.local_name}</h3>
          <em>${sp.botanical}</em>
          <p>Documented uses: ${formatDocumentedUses(sp)}</p>
        </div>
      </article>`
      )
      .join("");

    lib.querySelectorAll(".card-site").forEach((card) => {
      const clickHandler = () => {
        const slug = card.dataset.slug;
        const sp = speciesBySlug.get(slug);
        if (sp) openModal(sp);
      };
      card.addEventListener("click", clickHandler);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          clickHandler();
        }
      });
    });
  }

  q?.addEventListener("input", renderCatalogue);

  // File Upload and Analysis on Website
  const fileInput = document.getElementById("file");
  const dz = document.getElementById("dz");
  const goBtn = document.getElementById("go");
  const resetBtn = document.getElementById("reset");
  const res = document.getElementById("res");
  const dropLabel = document.getElementById("site-drop");

  let currentImageData = null;
  let currentSharpness = 0;
  let currentPreviewUrl = null;

  function clearSiteInput() {
    fileInput.value = "";
    dropLabel?.querySelector("img")?.remove();
    if (dz) dz.style.visibility = "visible";
    if (goBtn) goBtn.disabled = true;
    if (resetBtn) resetBtn.hidden = true;
    if (res) res.innerHTML = '<p class="empty">Results appear here.</p>';
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
      currentPreviewUrl = null;
    }
    currentImageData = null;
    currentSharpness = 0;
  }

  resetBtn?.addEventListener("click", clearSiteInput);

  async function handleSiteFile(file) {
    if (!file) return;
    try {
      const { imageData, sharpness } = await prepareImage(file);
      currentImageData = imageData;
      currentSharpness = sharpness;

      dropLabel?.querySelector("img")?.remove();
      const img = document.createElement("img");
      img.alt = "Specimen end-grain preview";
      if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl);
      currentPreviewUrl = URL.createObjectURL(file);
      img.src = currentPreviewUrl;
      dropLabel?.appendChild(img);
      if (dz) dz.style.visibility = "hidden";

      if (goBtn) goBtn.disabled = false;
      if (resetBtn) resetBtn.hidden = false;

      if (sharpness < MIN_SHARPNESS) {
        res.innerHTML = `<div class="note"><strong>Surface appears blurry (sharpness ${sharpness.toFixed(1)}).</strong> Minimum recommended sharpness is 100. Pores and rays must be sharp. You can replace the image or proceed anyway.</div>`;
      } else {
        res.innerHTML =
          '<p class="empty">Image ready. Confirm the pores and rays are sharp, then analyse.</p>';
      }
    } catch (err) {
      console.error(err);
      res.innerHTML = `<div class="note">${err.message || "Failed to load image."}</div>`;
      goBtn.disabled = true;
    }
  }

  fileInput?.addEventListener("change", () => {
    const f = fileInput.files?.[0];
    if (f) handleSiteFile(f);
  });

  // Drag and drop for website dropzone
  if (dropLabel) {
    dropLabel.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropLabel.classList.add("is-dragging");
    });
    dropLabel.addEventListener("dragleave", (e) => {
      if (!dropLabel.contains(e.relatedTarget)) {
        dropLabel.classList.remove("is-dragging");
      }
    });
    dropLabel.addEventListener("drop", (e) => {
      e.preventDefault();
      dropLabel.classList.remove("is-dragging");
      const f = e.dataTransfer?.files?.[0];
      if (f) handleSiteFile(f);
    });
  }

  goBtn?.addEventListener("click", async () => {
    if (!currentImageData) return;
    goBtn.disabled = true;
    res.innerHTML = `
      <p class="empty" style="font-family:var(--sans);font-weight:600">Analysing 48 regions…</p>
      <div class="bars"><i></i></div>
    `;

    try {
      const [{ session, config, classNames }] = await Promise.all([
        loadModel(),
        loadSpeciesData(),
      ]);

      const tensorValues = makeTensorData(
        currentImageData,
        config.normalization,
        GRID_ROWS,
        GRID_COLS,
        PATCH_SIZE
      );
      const tensor = new ort.Tensor("float32", tensorValues, [
        PATCH_COUNT,
        3,
        PATCH_SIZE,
        PATCH_SIZE,
      ]);

      const outputs = await session.run({ [session.inputNames[0]]: tensor });
      const logits = outputs[session.outputNames[0]]?.data;
      if (!logits || logits.length !== PATCH_COUNT * classNames.length) {
        throw new Error("Model inference produced invalid output dimensions.");
      }

      const result = aggregatePredictions(logits, classNames);
      const sp = result.winnerSpecies;

      // Save to on-device history
      addHistoryEntry({
        slug: result.winnerSlug,
        local_name: sp.local_name,
        botanical: sp.botanical,
        agreement: `${result.agreeingCount} of 48 regions`,
        date: new Date().toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
          year: "numeric",
        }),
      });

      // Render real result
      const gridHtml = Array.from(
        { length: 48 },
        (_, k) =>
          `<i class="${result.patchWinners[k] === result.winnerIndex ? "v" : ""}"></i>`
      ).join("");

      const topMatchesHtml = result.topMatches
        .map(
          (m, idx) => `
        <div class="cand">
          <span>${idx === 0 ? "Other top matches" : ""}</span>
          <b>${m.species?.local_name || m.slug} · ${m.count} regions</b>
        </div>`
        )
        .join("");

      res.innerHTML = `
        <div style="font:800 14px var(--sans);color:var(--muted);text-transform:uppercase;letter-spacing:0.05em">Most frequent match</div>
        <div class="big">${sp.local_name} <small>${sp.botanical}</small></div>
        <div style="margin:10px 0 6px">Patch agreement: <b style="font-family:var(--sans)">${result.agreeingCount} of 48 regions</b></div>
        <div class="grid48" aria-label="48-region grid: highlighted regions voted for ${sp.local_name}">${gridHtml}</div>
        ${topMatchesHtml}
        <div class="note">Patch agreement shows how consistent the regions were. It is not a probability and not a guarantee. Documented uses come from the catalogue, not from this image.</div>
        <div class="row">
          <button class="btn ghost" id="site-view-cat" style="color:var(--ink)">View in catalogue</button>
          <button class="btn ghost" id="site-report-btn" style="color:var(--ink)">Report incorrect result</button>
        </div>
      `;

      document.getElementById("site-view-cat")?.addEventListener("click", () => {
        openModal(sp);
      });

      const reportBtn = document.getElementById("site-report-btn");
      if (reportBtn) {
        reportBtn.addEventListener("click", () => {
          reportBtn.textContent = "Report noted (demo)";
          reportBtn.disabled = true;
        });
      }
    } catch (err) {
      console.error(err);
      res.innerHTML = `<div class="note" role="alert"><strong>Analysis error:</strong> ${err.message || "An error occurred during inference."}</div>`;
    } finally {
      goBtn.disabled = false;
    }
  });

  // Initial catalogue load
  loadSpeciesData()
    .then(() => renderCatalogue())
    .catch((err) => console.error("Error loading catalogue:", err));
}

/* ==========================================================================
   Mobile App Implementation (app.html)
   ========================================================================== */

function initMobileApp() {
  const logo = document.getElementById("logo");
  if (logo) {
    let s = 7;
    const rnd = () => (s = (s * 9301 + 49297) % 233280) / 233280;
    let h = "";
    for (let r = 92; r > 6; r -= 7) {
      h += `<circle r="${r}" fill="none" stroke="#A9B8A0" stroke-width="1" opacity=".6"/>`;
    }
    for (let i = 0; i < 70; i++) {
      const a = rnd() * 6.28,
        r = 10 + rnd() * 78;
      h += `<ellipse cx="${r * Math.cos(a)}" cy="${r * Math.sin(a)}" rx="${1 + rnd() * 1.6}" ry="${1 + rnd() * 1.6}" fill="#fff" opacity=".8"/>`;
    }
    logo.innerHTML = h;

    const splash = document.getElementById("splash");
    if (splash) {
      setTimeout(() => {
        splash.style.opacity = "0";
        setTimeout(() => splash.remove(), 500);
      }, 1400);
    }
  }

  const m = document.getElementById("m");
  const tabs = [...document.querySelectorAll(".tabs button")];

  let currentAppFile = null;
  let currentAppImageData = null;
  let currentAppSharpness = 0;
  let currentAppPreviewUrl = null;

  function setTab(tabName, html) {
    tabs.forEach((b) =>
      b.dataset.t === tabName
        ? b.setAttribute("aria-current", "page")
        : b.removeAttribute("aria-current")
    );
    if (!m) return;
    m.innerHTML = html;
    m.scrollTop = 0;
    m.focus();
    bindActions();
  }

  function bindActions() {
    m.querySelectorAll("[data-sp]").forEach((b) => {
      b.onclick = () => renderAppSpecies(b.dataset.sp);
    });
    m.querySelectorAll("[data-go]").forEach((b) => {
      b.onclick = () => navigateTo(b.dataset.go);
    });
  }

  function renderHistoryCard(item) {
    const sp = speciesBySlug.get(item.slug);
    const index = sp?.index ?? 0;
    return `
      <button class="card-app" data-sp="${item.slug}">
        <div class="sw">
          <img src="species_photos/${item.slug}.jpg" alt="${item.local_name}" onerror="this.style.display='none';this.parentElement.style.background='${cssGrain(grainColors[index % grainColors.length])}'">
        </div>
        <div class="t">
          <h3>${item.local_name}</h3>
          <em>${item.botanical}</em>
          <div class="small muted" style="margin-top:2px">${item.agreement} · ${item.date}</div>
        </div>
      </button>
    `;
  }

  function renderCatalogueCard(sp) {
    return `
      <button class="card-app" data-sp="${sp.slug}">
        <div class="sw">
          <img src="species_photos/${sp.slug}.jpg" alt="${sp.local_name}" loading="lazy" onerror="this.style.display='none';this.parentElement.style.background='${cssGrain(grainColors[sp.index % grainColors.length])}'">
        </div>
        <div class="t">
          <h3>${sp.local_name}</h3>
          <em>${sp.botanical}</em>
          <div class="small muted" style="margin-top:2px">${formatDocumentedUses(sp)}</div>
        </div>
      </button>
    `;
  }

  function screenHome() {
    const history = getHistory();
    const recent = history.slice(0, 2);
    setTab(
      "home",
      `
      <div class="kick">UNIVERSITY OF IBADAN</div>
      <h1>Examine the end-grain.</h1>
      <div class="bar"></div>
      <p class="muted" style="margin:0 0 18px">Compare a transverse cut surface against 29 Nigerian hardwood species. Analysis runs on this device.</p>
      <button class="btn red" data-go="cap">Identify a specimen</button>
      <h2 style="margin-top:30px">Recent</h2>
      ${
        recent.length
          ? recent.map(renderHistoryCard).join("")
          : '<p class="muted small">No identifications yet. Your results will be listed here.</p>'
      }
      <button class="btn ghost" data-go="set" style="margin-top:22px">About and settings</button>
      <a class="btn ghost" href="index.html?view=web" style="margin-top:10px">Switch to website view</a>
    `
    );
  }

  function screenCapture() {
    currentAppFile = null;
    currentAppImageData = null;
    if (currentAppPreviewUrl) {
      URL.revokeObjectURL(currentAppPreviewUrl);
      currentAppPreviewUrl = null;
    }

    setTab(
      "cap",
      `
      <h2>Photograph the cut face</h2>
      <p class="muted small" style="margin:0">Use the end-grain cross-section. JPEG or PNG, up to 8 MB. Never uploaded.</p>
      <div class="cap grain" style="--g:#33281c">
        <div class="frame"></div>
        <div class="hint">Fill the frame with wood. Hold square to the surface.</div>
      </div>
      <div class="two small">
        <div><b class="sans">Suitable</b><br>Flat, clean face, sharp, even light.</div>
        <div><b class="sans">Avoid</b><br>Side grain, bark, blur, glare.</div>
      </div>
      <label class="btn red" style="margin-top:16px;cursor:pointer">
        Use camera
        <input type="file" accept="image/*" capture="environment" id="app-f1" hidden>
      </label>
      <label class="btn ghost" style="margin-top:10px;cursor:pointer">
        Choose from files
        <input type="file" accept="image/jpeg,image/png" id="app-f2" hidden>
      </label>
      <div id="app-cap-error" style="margin-top:10px"></div>
    `
    );

    async function onFileSelected(e) {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        const { imageData, sharpness } = await prepareImage(file);
        currentAppFile = file;
        currentAppImageData = imageData;
        currentAppSharpness = sharpness;
        currentAppPreviewUrl = URL.createObjectURL(file);
        screenPreview();
      } catch (err) {
        const errDiv = document.getElementById("app-cap-error");
        if (errDiv) {
          errDiv.innerHTML = `<div class="note small" role="alert">${err.message || "Could not read image."}</div>`;
        }
      }
    }

    document.getElementById("app-f1")?.addEventListener("change", onFileSelected);
    document.getElementById("app-f2")?.addEventListener("change", onFileSelected);
  }

  function screenPreview() {
    setTab(
      "cap",
      `
      <button class="back" data-go="cap">&larr; Retake</button>
      <h2>Confirm image</h2>
      <div class="cap">
        <img src="${currentAppPreviewUrl}" alt="Specimen preview">
      </div>
      <p class="muted small">Check that pores and rays are sharp and evenly lit. Replace the image if not.</p>
      ${
        currentAppSharpness < MIN_SHARPNESS
          ? `<div class="note small" style="margin-bottom:12px">Image sharpness is ${currentAppSharpness.toFixed(1)} (min recommended 100). If blurry, retake for best results.</div>`
          : ""
      }
      <button class="btn red" id="app-analyse-btn" style="margin-top:8px">Analyse end-grain (48 regions)</button>
      <button class="btn ghost" data-go="cap" style="margin-top:10px">Replace image</button>
    `
    );

    document.getElementById("app-analyse-btn")?.addEventListener("click", runAnalysis);
  }

  async function runAnalysis() {
    setTab(
      "cap",
      `
      <div class="empty">
        <h2>Analysing 48 regions</h2>
        <p class="muted small">Classifying each patch, then taking a majority vote.</p>
        <div class="bars"><i></i></div>
      </div>
    `
    );

    try {
      const [{ session, config, classNames }] = await Promise.all([
        loadModel(),
        loadSpeciesData(),
      ]);

      const tensorValues = makeTensorData(
        currentAppImageData,
        config.normalization,
        GRID_ROWS,
        GRID_COLS,
        PATCH_SIZE
      );
      const tensor = new ort.Tensor("float32", tensorValues, [
        PATCH_COUNT,
        3,
        PATCH_SIZE,
        PATCH_SIZE,
      ]);

      const outputs = await session.run({ [session.inputNames[0]]: tensor });
      const logits = outputs[session.outputNames[0]]?.data;
      if (!logits || logits.length !== PATCH_COUNT * classNames.length) {
        throw new Error("Model inference produced invalid output dimensions.");
      }

      const result = aggregatePredictions(logits, classNames);
      const sp = result.winnerSpecies;

      // Save to on-device history
      addHistoryEntry({
        slug: result.winnerSlug,
        local_name: sp.local_name,
        botanical: sp.botanical,
        agreement: `${result.agreeingCount} of 48 regions`,
        date: new Date().toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
          year: "numeric",
        }),
      });

      const gridHtml = Array.from(
        { length: 48 },
        (_, k) =>
          `<i class="${result.patchWinners[k] === result.winnerIndex ? "v" : ""}"></i>`
      ).join("");

      const topMatchesHtml = result.topMatches
        .map(
          (m, idx) => `
        <div class="row-app">
          <span>${idx === 0 ? "Other top matches" : ""}</span>
          <b>${m.species?.local_name || m.slug} · ${m.count}</b>
        </div>`
        )
        .join("");

      setTab(
        "cap",
        `
        <div class="kick" style="margin-top:14px">MOST FREQUENT MATCH</div>
        <div class="big">${sp.local_name}</div>
        <em class="muted">${sp.botanical}</em>
        <p style="margin:12px 0 0"><b class="sans">Patch agreement: ${result.agreeingCount} of 48 regions</b></p>
        <div class="g48" aria-label="48-region grid: highlighted regions voted for ${sp.local_name}">${gridHtml}</div>
        ${topMatchesHtml}
        <div class="note small">Patch agreement shows consistency between regions. It is not a probability or a guarantee. Documented uses come from the catalogue, not from your photo.</div>
        <button class="btn" data-sp="${sp.slug}" style="margin-top:16px">View species profile</button>
        <button class="btn ghost" id="app-report-btn" style="margin-top:10px">Report incorrect result</button>
        <button class="btn ghost" data-go="cap" style="margin-top:10px">New specimen</button>
      `
      );

      const reportBtn = document.getElementById("app-report-btn");
      if (reportBtn) {
        reportBtn.addEventListener("click", () => {
          reportBtn.textContent = "Report noted (demo)";
          reportBtn.disabled = true;
        });
      }
    } catch (err) {
      console.error(err);
      setTab(
        "cap",
        `
        <div class="note small" role="alert">
          <strong>Analysis error:</strong> ${err.message || "Inference failed."}
        </div>
        <button class="btn red" data-go="cap" style="margin-top:16px">Try again</button>
      `
      );
    }
  }

  function screenCatalogue() {
    setTab(
      "lib",
      `
      <h2>Species catalogue</h2>
      <p class="muted small" style="margin:0">Names and uses come from literature and database records for 29 Ibadan hardwoods.</p>
      <input class="search" id="app-q" type="search" placeholder="Search common or scientific name" aria-label="Search species">
      <div id="app-lib-list"></div>
    `
    );

    const listDiv = document.getElementById("app-lib-list");
    const qInput = document.getElementById("app-q");

    function renderList() {
      const term = (qInput?.value || "").trim().toLowerCase();
      const filtered = speciesList.filter((sp) =>
        `${sp.local_name} ${sp.botanical} ${sp.english_name || ""}`
          .toLowerCase()
          .includes(term)
      );

      if (!filtered.length) {
        listDiv.innerHTML =
          '<div class="empty muted">No species match. Try a common name or genus.</div>';
      } else {
        listDiv.innerHTML = filtered.map(renderCatalogueCard).join("");
      }
      bindActions();
    }

    qInput?.addEventListener("input", renderList);
    renderList();
  }

  function renderAppSpecies(slug) {
    const sp = speciesBySlug.get(slug);
    if (!sp) return;
    const uses = sp.documented_uses ?? sp.uses ?? {};
    const entries = Object.entries(uses);
    const usesFormatted = entries.length
      ? entries.map(([u]) => u.replaceAll("_", " ")).join(", ")
      : "None documented in current database";

    setTab(
      "lib",
      `
      <button class="back" data-go="lib">&larr; Catalogue</button>
      <div class="sw" style="height:190px;overflow:hidden;border-radius:2px">
        <img src="species_photos/${sp.slug}.jpg" alt="${sp.local_name}" style="width:100%;height:100%;object-fit:cover" onerror="this.style.display='none';this.parentElement.style.background='${cssGrain(grainColors[sp.index % grainColors.length])}'">
      </div>
      <h1 style="font-size:32px">${sp.local_name}</h1>
      <em class="muted">${sp.botanical}</em>
      <div class="bar"></div>
      <dl style="margin:0">
        <dt>Wood type</dt>
        <dd>${sp.wood_type || "Hardwood"}</dd>
        <dt>Documented applications</dt>
        <dd>${usesFormatted}</dd>
        <dt>Source</dt>
        <dd>Literature and database records. Not predicted from an image.</dd>
      </dl>
      <button class="btn red" data-go="cap" style="margin-top:20px">Identify a specimen</button>
    `
    );
  }

  function screenHistory() {
    const history = getHistory();
    setTab(
      "hist",
      `
      <h2>History</h2>
      ${
        history.length
          ? history.map(renderHistoryCard).join("") +
            '<button class="btn ghost" id="app-clear-hist" style="margin-top:18px">Clear history</button>'
          : '<div class="empty"><h3>No identifications yet</h3><p class="muted small">Photograph an end-grain face and the result will be saved here.</p><button class="btn red" data-go="cap">Identify a specimen</button></div>'
      }
    `
    );

    document.getElementById("app-clear-hist")?.addEventListener("click", () => {
      clearHistory();
      screenHistory();
    });
  }

  function screenSettings() {
    setTab(
      "home",
      `
      <button class="back" data-go="home">&larr; Home</button>
      <h2>About and settings</h2>
      <dl style="margin:0">
        <dt>Scope</dt>
        <dd>29 hardwood species traded in Ibadan wood markets, on transverse end-grain surfaces. Softwoods and other species are not recognised.</dd>
        <dt>Privacy</dt>
        <dd>Images are analysed on this device and are not uploaded.</dd>
        <dt>Appropriate use</dt>
        <dd>Not for legal, regulatory, purchasing, import/export or structural-grading decisions.</dd>
      </dl>
      <details>
        <summary>Read full limitations and evaluation</summary>
        <p class="muted small">Reported independent-specimen image-level accuracy was 12.1% (ShuffleNetV2, primary) and 11.3% (ResNet18) on 611 held-out images from 14 species. Fifteen single-specimen species lacked independent test results. Results apply only to the defined scope.</p>
      </details>
      <button class="btn ghost" id="app-clear-hist2" style="margin-top:16px">Clear saved history</button>
      <a class="btn ghost" href="index.html?view=web" style="margin-top:10px">Switch to website view</a>
    `
    );

    const clr = document.getElementById("app-clear-hist2");
    if (clr) {
      clr.onclick = () => {
        clearHistory();
        clr.textContent = "History cleared";
        clr.disabled = true;
      };
    }
  }

  function navigateTo(target) {
    if (target === "home") screenHome();
    else if (target === "cap") screenCapture();
    else if (target === "lib") screenCatalogue();
    else if (target === "hist") screenHistory();
    else if (target === "set") screenSettings();
  }

  tabs.forEach((b) => {
    b.addEventListener("click", () => navigateTo(b.dataset.t));
  });

  // Start at home screen
  loadSpeciesData()
    .then(() => screenHome())
    .catch((err) => console.error("Error loading species in mobile app:", err));
}

/* ==========================================================================
   Page Initialization
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("acquire") || document.getElementById("fig")) {
    initWebsite();
  }
  if (document.getElementById("app")) {
    initMobileApp();
  }
});
