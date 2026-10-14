// Runs the Python package with Pyodide in a background thread, so the animation stays smooth.
// Pyodide 314 needs a module worker (see startPython in app.js).
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

async function loadPython() {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage("numpy");
  const archive = await fetch("py/lab_in_the_loop.zip");
  if (!archive.ok) throw new Error("Could not download the Python package.");
  pyodide.unpackArchive(await archive.arrayBuffer(), "zip");
  return pyodide.pyimport("lab_in_the_loop.browser");
}

const python = loadPython();

self.onmessage = async ({ data }) => {
  try {
    const browser = await python;
    const result = data.type === "step"
      ? JSON.parse(browser.step(data.seed, JSON.stringify(data.history)))
      : null;
    self.postMessage({ id: data.id, result });
  } catch (error) {
    self.postMessage({ id: data.id, error: String(error) });
  }
};
