"use strict";

// Must match N_SCREEN_EXPERIMENTS in lab_in_the_loop/loop.py.
const screenExperiments = 5;
const N_CELLS = 10;
const N_COMPOUNDS = 10;
const N_HIDDEN_SHOWN = 12;
const N_HIDDEN = 32;
const PHENOTYPES = [
  { name: "Colour", short: "Colour", ends: ["red", "green"] },
  { name: "Shape", short: "Shape", ends: ["round", "square"] },
  { name: "Nucleus size", short: "Nucleus", ends: ["small", "big"] },
];
const METHOD_LABELS = { start: "start", screen: "random screen", model: "model" };

// The page keeps the whole session. Python keeps nothing between calls.
// history: concentrations applied so far. next: the proposal for the next iteration.
const state = { seed: 0, history: [], results: [], next: null, ready: false, busy: false };

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const errorOf = (phenotype) => phenotype.reduce((sum, v) => sum + Math.abs(1 - v), 0);
const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attributes = {}, parent = null) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (parent) parent.appendChild(element);
  return element;
}

// Small seeded random generator, so the dish layout is the same on every load.
function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- Python in the browser ---------- */

// The Python package runs in a Web Worker (worker.js). Each call gets an id,
// so the answer can find the promise that waits for it.
let worker = null;
let nextMessageId = 0;
const pending = new Map();

function callPython(message) {
  const id = nextMessageId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, ...message });
  });
}

function startPython() {
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    const { resolve, reject } = pending.get(data.id);
    pending.delete(data.id);
    if (data.error) reject(new Error(data.error.trim().split("\n").pop()));
    else resolve(data.result);
  };
  worker.onerror = (event) => {
    for (const { reject } of pending.values()) reject(new Error(event.message || "Python could not start."));
    pending.clear();
  };
  return callPython({ type: "load" });
}

function postStep(history) {
  return callPython({ type: "step", seed: state.seed, history });
}

/* ---------- Status, buttons and stepper ---------- */

function setStatus(text, busy = false) {
  $("status").textContent = text;
  $("status").classList.toggle("busy", busy);
}

function setBusy(busy) {
  state.busy = busy;
  $("start").disabled = busy || state.ready;
  $("step").disabled = busy || !state.ready || !state.next;
  $("reset").disabled = busy || !state.ready;
  $("seed").disabled = busy;
}

function setStepper(activeStep) {
  for (const item of document.querySelectorAll(".stepper li")) {
    const step = Number(item.dataset.step);
    item.classList.toggle("active", step === activeStep);
    item.classList.toggle("done", activeStep !== null && step < activeStep);
  }
}

/* ---------- Step 1: the dish ---------- */

const cells = [];

function buildDish() {
  const random = mulberry32(42);
  const positions = [];
  for (let attempt = 0; attempt < 5000 && positions.length < N_CELLS; attempt++) {
    const angle = random() * 2 * Math.PI;
    const radius = Math.sqrt(random()) * 35;
    const x = 50 + radius * Math.cos(angle);
    const y = 50 + radius * Math.sin(angle);
    if (positions.every((p) => Math.hypot(p.x - x, p.y - y) > 18)) positions.push({ x, y });
  }
  for (const { x, y } of positions) {
    const size = 12 + random() * 3;
    const element = document.createElement("div");
    element.className = "cell";
    Object.assign(element.style, {
      left: `${x - size / 2}%`, top: `${y - size / 2}%`, width: `${size}%`, height: `${size}%`,
      transform: `rotate(${random() * 90}deg)`,
    });
    const nucleus = document.createElement("div");
    nucleus.className = "nucleus";
    element.appendChild(nucleus);
    $("dish").appendChild(element);
    // Each cell varies a little around the measured phenotype, as real cells do.
    cells.push({ element, nucleus, jitter: [0, 1, 2].map(() => (random() - 0.5) * 0.2) });
  }
}

function cellColour(value) {
  const mix = (other, amount) =>
    `color-mix(in oklab, var(${other}) ${Math.round(amount * 100)}%, var(--cell-blue))`;
  return value < 0 ? mix("--cell-red", -value) : mix("--cell-green", value);
}

function renderCells(phenotype) {
  $("dish").classList.remove("is-empty");
  for (const cell of cells) {
    const [colour, shape, nucleus] = phenotype.map((v, i) => clamp(v + cell.jitter[i], -1, 1));
    cell.element.style.backgroundColor = cellColour(colour);
    cell.element.style.borderRadius = `${50 - ((shape + 1) / 2) * 36}%`;
    const nucleusSize = `${22 + ((nucleus + 1) / 2) * 40}%`;
    cell.nucleus.style.width = nucleusSize;
    cell.nucleus.style.height = nucleusSize;
  }
  const values = phenotype.map((v, i) => `${PHENOTYPES[i].name.toLowerCase()} ${v.toFixed(2)}`);
  $("dish-caption").textContent = `Measured: ${values.join(", ")}.`;
}

/* ---------- Step 2: the error panel ---------- */

const meters = [];

function buildPhenotypes() {
  for (const { name, ends } of PHENOTYPES) {
    const block = document.createElement("div");
    block.innerHTML =
      '<div class="meter-label"><span class="ends"></span><strong></strong><span class="distance"></span><span class="ends"></span></div>' +
      '<div class="meter"><i class="tick"></i><i class="dot predicted hidden"></i><i class="dot measured hidden"></i></div>';
    const [left, right] = block.querySelectorAll(".ends");
    left.textContent = ends[0];
    right.textContent = ends[1];
    block.querySelector("strong").textContent = name;
    $("phenotypes").appendChild(block);
    meters.push({
      measured: block.querySelector(".measured"),
      predicted: block.querySelector(".predicted"),
      distance: block.querySelector(".distance"),
    });
  }
}

function placeDot(dot, value) {
  dot.classList.toggle("hidden", value === null);
  if (value !== null) dot.style.left = `${((value + 1) / 2) * 100}%`;
}

function renderError(measurement, predictedPhenotype) {
  const panel = $("error-panel");
  panel.classList.remove("is-waiting", "reveal");
  void panel.offsetWidth; // Restart the reveal animation.
  panel.classList.add("reveal");
  $("error-value").textContent = measurement.loss.toFixed(2);
  $("error-predicted").textContent = predictedPhenotype
    ? `Before the experiment, the model predicted ${errorOf(predictedPhenotype).toFixed(2)}.`
    : "No model prediction for this experiment: it came from the start or the random screen.";
  meters.forEach((meter, i) => {
    placeDot(meter.measured, measurement.phenotype[i]);
    placeDot(meter.predicted, predictedPhenotype ? predictedPhenotype[i] : null);
    meter.distance.textContent = `off by ${Math.abs(1 - measurement.phenotype[i]).toFixed(2)}`;
  });
}

function clearError() {
  $("error-panel").classList.add("is-waiting");
  $("error-value").textContent = "n/a";
  $("error-predicted").textContent = "Sum of distances from the target. 0 is perfect, 6 is the worst.";
  meters.forEach((meter) => {
    placeDot(meter.measured, null);
    placeDot(meter.predicted, null);
    meter.distance.textContent = "";
  });
}

/* ---------- Step 3: the network ---------- */

const network = { inputHidden: [], hiddenOutput: [], shownInputHidden: null, shownHiddenOutput: null };

function zeros(rows, columns) {
  return Array.from({ length: rows }, () => new Array(columns).fill(0));
}

function buildNetwork() {
  const width = 340;
  const height = 262;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Neural network weights" }, $("network"));
  const column = { input: 44, hidden: 170, output: 280 };
  const spread = (n, top, bottom) => Array.from({ length: n }, (_, i) => top + (i * (bottom - top)) / (n - 1));
  const inputY = spread(N_COMPOUNDS, 32, 232);
  const hiddenY = spread(N_HIDDEN_SHOWN, 32, 232);
  const outputY = spread(PHENOTYPES.length, 76, 188);

  svg("text", { class: "layer-label", x: column.input, y: 14, "text-anchor": "middle" }, root).textContent = "Concentrations";
  svg("text", { class: "layer-label", x: column.hidden, y: 14, "text-anchor": "middle" }, root).textContent = `Hidden (${N_HIDDEN_SHOWN} of ${N_HIDDEN})`;
  svg("text", { class: "layer-label", x: column.output, y: 14, "text-anchor": "middle" }, root).textContent = "Phenotype";

  const edges = svg("g", { class: "edges" }, root);
  for (let h = 0; h < N_HIDDEN_SHOWN; h++) {
    network.inputHidden.push(inputY.map((y) => svg("line", { x1: column.input, y1: y, x2: column.hidden, y2: hiddenY[h] }, edges)));
  }
  for (let o = 0; o < PHENOTYPES.length; o++) {
    network.hiddenOutput.push(hiddenY.map((y) => svg("line", { x1: column.hidden, y1: y, x2: column.output, y2: outputY[o] }, edges)));
  }
  inputY.forEach((y, i) => {
    svg("circle", { class: "node input-node", cx: column.input, cy: y, r: 7 }, root);
    svg("text", { class: "label", x: column.input - 12, y: y + 4, "text-anchor": "end" }, root).textContent = `C${i + 1}`;
  });
  hiddenY.forEach((y) => svg("circle", { class: "node hidden-node", cx: column.hidden, cy: y, r: 5 }, root));
  outputY.forEach((y, o) => {
    svg("circle", { class: "node output-node", cx: column.output, cy: y, r: 8 }, root);
    svg("text", { class: "label", x: column.output + 13, y: y + 4 }, root).textContent = PHENOTYPES[o].short;
  });
  svg("text", { class: "lock", x: column.hidden, y: height - 4, "text-anchor": "middle" }, root).textContent = "🔒 Weights frozen";

  network.shownInputHidden = zeros(N_HIDDEN_SHOWN, N_COMPOUNDS);
  network.shownHiddenOutput = zeros(PHENOTYPES.length, N_HIDDEN_SHOWN);
  drawWeights(1, 1);
}

function styleEdge(line, weight, scale) {
  const strength = clamp(Math.abs(weight) / scale, 0, 1);
  line.setAttribute("stroke", weight >= 0 ? "var(--series-1)" : "var(--negative)");
  line.setAttribute("stroke-width", (0.3 + 2.7 * strength).toFixed(2));
  line.setAttribute("stroke-opacity", (0.08 + 0.8 * strength).toFixed(2));
}

function drawWeights(inputScale, outputScale) {
  network.inputHidden.forEach((row, h) =>
    row.forEach((line, i) => styleEdge(line, network.shownInputHidden[h][i], inputScale)));
  network.hiddenOutput.forEach((row, o) =>
    row.forEach((line, h) => styleEdge(line, network.shownHiddenOutput[o][h], outputScale)));
}

function maxAbs(matrix) {
  return Math.max(1e-6, ...matrix.flat().map(Math.abs));
}

// Move the drawn weights from their old values to the new ones.
function animateWeights(model, durationMs) {
  const targetInput = model.input_hidden_weights.slice(0, N_HIDDEN_SHOWN);
  const targetOutput = model.hidden_output_weights.map((row) => row.slice(0, N_HIDDEN_SHOWN));
  const startInput = network.shownInputHidden.map((row) => row.slice());
  const startOutput = network.shownHiddenOutput.map((row) => row.slice());
  const inputScale = maxAbs(targetInput);
  const outputScale = maxAbs(targetOutput);
  const lerp = (from, to, t) => from.map((row, r) => row.map((v, c) => v + (to[r][c] - v) * t));

  return new Promise((resolve) => {
    const startTime = performance.now();
    function frame(now) {
      const t = clamp((now - startTime) / durationMs, 0, 1);
      const eased = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      network.shownInputHidden = lerp(startInput, targetInput, eased);
      network.shownHiddenOutput = lerp(startOutput, targetOutput, eased);
      drawWeights(inputScale, outputScale);
      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });
}

function setNetworkMode(mode, tagText) {
  $("network").classList.toggle("training", mode === "training");
  $("network").classList.toggle("frozen", mode === "frozen");
  $("model-tag").textContent = tagText;
  $("model-tag").className = `tag ${mode ?? ""}`;
}

function resetNetwork() {
  network.shownInputHidden = zeros(N_HIDDEN_SHOWN, N_COMPOUNDS);
  network.shownHiddenOutput = zeros(PHENOTYPES.length, N_HIDDEN_SHOWN);
  drawWeights(1, 1);
  setNetworkMode(null, "Not trained");
  $("model-note").textContent =
    "The model is a small feed-forward net: 10 concentrations in, 32 hidden units, 3 phenotypes out. The drawing shows 12 of the 32 hidden units.";
}

/* ---------- Step 4: concentration bars ---------- */

const compoundRows = [];

function buildCompounds() {
  for (let i = 0; i < N_COMPOUNDS; i++) {
    const row = document.createElement("div");
    row.className = "compound";
    row.innerHTML = '<span class="name"></span><div class="track"><div class="fill"></div><i class="tick hidden"></i></div><span class="value"></span>';
    row.querySelector(".name").textContent = `C${i + 1}`;
    $("compounds").appendChild(row);
    compoundRows.push({ fill: row.querySelector(".fill"), tick: row.querySelector(".tick"), value: row.querySelector(".value") });
  }
}

function renderCompounds(concentrations, previous) {
  compoundRows.forEach((row, i) => {
    const value = concentrations ? concentrations[i] : 0;
    row.fill.style.width = `${value * 100}%`;
    row.value.textContent = concentrations ? value.toFixed(3) : "";
    row.tick.classList.toggle("hidden", !previous);
    if (previous) row.tick.style.left = `${previous[i] * 100}%`;
  });
}

function setCompoundNote(text, highlight = false) {
  $("compound-note").textContent = text;
  $("compound-note").classList.toggle("highlight", highlight);
}

/* ---------- Error chart ---------- */

function renderChart() {
  const container = $("chart");
  container.replaceChildren();
  const width = container.clientWidth;
  const height = 170;
  const margin = { left: 32, right: 48, top: 16, bottom: 28 };
  const results = state.results;
  const xMax = Math.max(9, results.length - 1);
  const x = (i) => margin.left + (i / xMax) * (width - margin.left - margin.right);
  const y = (loss) => margin.top + (1 - loss / 6) * (height - margin.top - margin.bottom);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Error for each experiment" }, container);

  const bandEnd = x(Math.min(screenExperiments - 0.5, xMax));
  svg("rect", { class: "band", x: margin.left, y: margin.top, width: bandEnd - margin.left, height: height - margin.top - margin.bottom }, root);
  svg("text", { class: "axis-label", x: margin.left + 6, y: margin.top + 14 }, root).textContent = "Start and random screen";
  svg("text", { class: "axis-label", x: bandEnd + 6, y: margin.top + 14 }, root).textContent = "Chosen by the model";

  for (const tick of [0, 2, 4, 6]) {
    svg("line", { class: "grid-line", x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick) }, root);
    svg("text", { class: "axis-label", x: margin.left - 8, y: y(tick) + 4, "text-anchor": "end" }, root).textContent = tick;
  }
  const xStep = Math.ceil(xMax / 10);
  for (let i = 0; i <= xMax; i += xStep) {
    svg("text", { class: "axis-label", x: x(i), y: height - 8, "text-anchor": "middle" }, root).textContent = i;
  }
  if (results.length === 0) return;

  svg("polyline", { class: "line", points: results.map((r, i) => `${x(i)},${y(r.loss)}`).join(" ") }, root);
  results.forEach((r, i) => svg("circle", { class: "marker", cx: x(i), cy: y(r.loss), r: 4 }, root));
  const last = results[results.length - 1];
  svg("text", { class: "end-label", x: x(results.length - 1) + 8, y: y(last.loss) + 4 }, root).textContent = last.loss.toFixed(2);

  // Hover layer: a crosshair snaps to the nearest experiment.
  const crosshair = svg("line", { class: "crosshair", y1: margin.top, y2: height - margin.bottom, visibility: "hidden" }, root);
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  tooltip.hidden = true;
  container.appendChild(tooltip);
  const hit = svg("rect", { x: margin.left, y: 0, width: width - margin.left - margin.right, height, fill: "transparent" }, root);
  hit.addEventListener("pointermove", (event) => {
    const box = root.getBoundingClientRect();
    const fraction = (event.clientX - box.left - margin.left) / (width - margin.left - margin.right);
    const index = clamp(Math.round(fraction * xMax), 0, results.length - 1);
    const result = results[index];
    crosshair.setAttribute("x1", x(index));
    crosshair.setAttribute("x2", x(index));
    crosshair.setAttribute("visibility", "visible");
    const value = document.createElement("strong");
    value.textContent = `Error ${result.loss.toFixed(2)}`;
    const label = document.createElement("span");
    label.textContent = `Experiment ${index}, ${METHOD_LABELS[result.method]}`;
    tooltip.replaceChildren(value, label);
    tooltip.hidden = false;
    tooltip.style.left = `${clamp(x(index) + 12, 0, width - 170)}px`;
    tooltip.style.top = `${y(result.loss) - 52}px`;
  });
  hit.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    tooltip.hidden = true;
  });
}

function addTableRow(result) {
  const row = document.createElement("tr");
  for (const text of [result.experiment_index, METHOD_LABELS[result.method], result.loss.toFixed(3)]) {
    const cell = document.createElement("td");
    cell.textContent = text;
    row.appendChild(cell);
  }
  $("table-body").appendChild(row);
}

/* ---------- The loop ---------- */

async function loadStartConcentrations() {
  const result = await postStep([]);
  state.next = result.proposal;
  renderCompounds(state.next.concentrations, null);
  setCompoundNote("Start: all concentrations near zero, so the first iteration measures the cells as they are now.", true);
  setStatus("Ready. Click 'Perform lab-in-the-loop iteration'.");
}

async function runIteration() {
  const applying = state.next;
  const history = [...state.history, applying.concentrations];
  const request = postStep(history);

  // Step 1: the lab applies the concentrations chosen in the previous iteration.
  setStepper(1);
  setStatus(`Applying compounds to the cells (experiment ${applying.experiment_index})...`, true);
  $("experiment-label").textContent = `Experiment ${applying.experiment_index}`;
  setCompoundNote("These concentrations are now applied to the cells.");
  setNetworkMode(null, $("model-tag").textContent === "Not trained" ? "Not trained" : "Waiting");
  $("dish").classList.add("incubating");
  const [result] = await Promise.all([request, sleep(1100)]);

  state.history = history;
  state.results.push({ experiment_index: applying.experiment_index, loss: result.measurement.loss, method: applying.method });
  state.next = result.proposal;

  $("dish").classList.remove("incubating");
  renderCells(result.measurement.phenotype);
  await sleep(1400);

  // Step 2: quantify the error next to the cells.
  setStepper(2);
  setStatus("Measuring the error...", true);
  renderError(result.measurement, applying.predicted_phenotype);
  renderChart();
  addTableRow(state.results[state.results.length - 1]);
  await sleep(2000);

  // Step 3: the model updates its weights to reflect the lab work.
  setStepper(3);
  setStatus("Updating the model weights...", true);
  setNetworkMode("training", "Training");
  const n = result.model.n_experiments;
  $("model-note").textContent = `Updating the weights: the model trains on all ${n} lab result${n === 1 ? "" : "s"}, so it now reflects the lab work.`;
  await animateWeights(result.model, 1800);
  $("model-note").textContent += ` Fit error (mean squared): ${result.model.fit_error.toFixed(4)}.`;
  await sleep(1200);

  // Step 4: freeze the model and update the concentrations.
  setStepper(4);
  setStatus("Model frozen. Choosing new concentrations...", true);
  setNetworkMode("frozen", "Frozen");
  const usesModel = state.next.method === "model";
  $("model-note").textContent = usesModel
    ? "Weights frozen. Now the concentrations are the trainable values: gradient descent through the frozen model moves them toward a lower predicted error."
    : `Weights frozen. With fewer than ${screenExperiments} lab results the model is not trusted yet, so the next concentrations come from a random screen.`;
  renderCompounds(state.next.concentrations, applying.concentrations);
  const prediction = usesModel ? ` The model predicts an error of ${errorOf(state.next.predicted_phenotype).toFixed(2)}.` : "";
  setCompoundNote(`New concentrations. They will be used in the next iteration.${prediction}`, true);
  await sleep(1200);

  setStepper(5);
  const best = Math.min(...state.results.map((r) => r.loss));
  setStatus(`Iteration done. Error ${result.measurement.loss.toFixed(2)}, best so far ${best.toFixed(2)}.`);
}

async function guarded(action) {
  setBusy(true);
  try {
    await action();
  } catch (error) {
    $("dish").classList.remove("incubating");
    setStatus(`Error: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

function resetSession() {
  state.history = [];
  state.results = [];
  state.next = null;
  $("table-body").replaceChildren();
  $("dish").classList.add("is-empty");
  $("dish-caption").textContent = "No cells yet.";
  $("experiment-label").textContent = "";
  renderCompounds(null, null);
  setCompoundNote("From 0 to 1. A sigmoid keeps every value above 0 and below 1.");
  clearError();
  resetNetwork();
  setStepper(null);
  renderChart();
}

async function start() {
  setStatus("Loading Python and NumPy into your browser (about 10 MB; fast after the first time)...", true);
  await startPython();
  state.ready = true;
  await loadStartConcentrations();
}

function init() {
  state.seed = Math.floor(Math.random() * 100000);
  $("seed").value = state.seed;
  buildDish();
  buildPhenotypes();
  buildNetwork();
  buildCompounds();
  resetSession();

  $("start").addEventListener("click", () => guarded(start));
  $("step").addEventListener("click", () => guarded(runIteration));
  $("reset").addEventListener("click", () => {
    resetSession();
    guarded(loadStartConcentrations);
  });
  $("seed").addEventListener("change", () => {
    const seed = Number.parseInt($("seed").value, 10);
    state.seed = Number.isInteger(seed) ? clamp(seed, 0, 2147483647) : 0;
    $("seed").value = state.seed;
    resetSession();
    if (state.ready) guarded(loadStartConcentrations);
  });
  window.addEventListener("resize", renderChart);
}

init();
