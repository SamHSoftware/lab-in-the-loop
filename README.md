# Lab in the loop

![Licence](https://img.shields.io/badge/Licence-MIT-blue)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SamHSoftware/a3a89be5685fb1b48eac281025e4c214/raw/coverage.json)

A small demo of lab-in-the-loop optimisation. A fictitious lab has blue, round cells with
small nuclei and wants green, square cells with big nuclei. Ten compounds have unknown effects.
A tiny neural net learns from each experiment and proposes the next one. Everything runs in
the browser: the Python code runs in [Pyodide](https://pyodide.org) (Python compiled to WebAssembly).

## How it works

Each click on "Perform lab-in-the-loop iteration" runs four steps:

1. **Apply compounds.** The fictitious lab applies the concentrations chosen in the
   previous iteration, and the cells change.
2. **Measure the error.** The error is the distance from the target phenotype, summed
   over colour, shape and nucleus size (0 is perfect, 6 is the worst).
3. **Update the model.** Train an ensemble of 5 small feed-forward nets on all lab results
   so far (compound concentrations in, measured phenotype out).
4. **Freeze and propose.** Freeze the weights. Run gradient descent on the *concentrations*
   to reduce the predicted error, minus a bonus where the nets disagree (exploration).
   Concentrations are stored as logits and passed through a sigmoid, so they never reach
   0 or 1. These concentrations are used in the next iteration.

The first 5 experiments do not use the model's proposals: experiment 0 measures the
current cells, and experiments 1 to 4 are a random screen, so the model has data to learn from.

Why not backpropagate the error through the lab? A real lab is a black box with no
gradients. The model is a differentiable stand-in (a surrogate) for the lab, and the
gradients flow through the model.

| Module | Role |
|---|---|
| `lab_in_the_loop/simulator.py` | The hidden biology. The seed fixes the effect of each compound. |
| `lab_in_the_loop/surrogate.py` | The ensemble in NumPy, with hand-written backpropagation, and the proposal step. |
| `lab_in_the_loop/loop.py` | One iteration of the loop. |
| `lab_in_the_loop/browser.py` | The JSON entry point which the page calls through Pyodide. |
| `web/` | The page. `worker.js` runs Pyodide in a background thread. |
| `scripts/build_site.py` | Copies `web/` to `site/` and zips the Python package for the browser. |

The model is NumPy, not PyTorch, because Pyodide has no PyTorch. The backpropagation is
written by hand, and tests check it against finite-difference gradients.

The page keeps the session: the seed and the list of applied concentrations (newest last).
Each iteration sends them to Python, which recomputes the lab results and retrains from zero
(about 70 ms in the browser). An empty list returns only the start concentrations.

## Local development

```sh
micromamba create -f environment.yml
micromamba activate python-repo-template
poetry install --with dev
pre-commit install -t pre-commit
pre-commit install -t pre-push
```

Build the site and serve it, then open <http://localhost:8000>:

```sh
python scripts/build_site.py
python -m http.server 8000 -d site
```

Run the build again after each change to `web/` or `lab_in_the_loop/`.

Local Python is 3.12 and the browser runs Python 3.14 (Pyodide 314.0.7). NumPy is pinned to
the same version as Pyodide's (2.4.6). The two give the same results.

## Deployment

GitHub Pages, from `.github/workflows/pages.yml`. It builds `site/` and publishes it on each
push to `main` which changes `web/`, `lab_in_the_loop/` or the build script.

One-time setup: in the repo settings, set **Pages → Source** to **GitHub Actions**.
There is no server, so there is no running cost.
