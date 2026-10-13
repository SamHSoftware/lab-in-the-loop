# Lab in the loop

![Licence](https://img.shields.io/badge/Licence-MIT-blue)

A small demo of lab-in-the-loop optimisation. A fictitious lab has blue, round cells with
small nuclei and wants green, square cells with big nuclei. Ten compounds have unknown effects.
A tiny PyTorch model learns from each experiment and proposes the next one.

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
| `lab_in_the_loop/surrogate.py` | The ensemble and the proposal step. |
| `lab_in_the_loop/loop.py` | One turn of the loop. |
| `lab_in_the_loop/api.py` | FastAPI: `GET /health` and `POST /step`. |
| `web/` | Static page with the animation. |

The server is stateless. The browser keeps the seed and the list of applied concentrations
(newest last) and sends them with each iteration. The server recomputes the lab results and
retrains from zero (about 0.5 s). An empty list returns only the start concentrations.

## Local development

```sh
micromamba create -f environment.yml
micromamba activate python-repo-template
poetry install --with dev
pre-commit install -t pre-commit
pre-commit install -t pre-push
```

Run the API and the page in two terminals, then open <http://localhost:8000>:

```sh
uvicorn lab_in_the_loop.api:app --port 7860
python -m http.server 8000 -d web
```

## Deployment

- **Web page:** GitHub Pages, from `web/` (`.github/workflows/pages.yml`).
- **API:** a Hugging Face Docker Space on the free CPU tier. CI builds the image, pushes it to
  GHCR, and writes a one-line `Dockerfile` (`FROM ghcr.io/...@sha256:...`) to the Space
  (`.github/workflows/publish-docker.yml`).

One-time setup:

1. Create a Hugging Face account **with no payment method**, and a public Docker Space.
2. Create a fine-grained HF token with write access to that Space only.
3. In the GitHub repo settings:
   - Add the secret `HF_TOKEN`.
   - Add the variable `HF_SPACE_ID`, for example `your-hf-name/lab-in-the-loop`.
   - Set Pages to deploy from "GitHub Actions".
4. After the first image push, make the GHCR package public, so that HF can pull it.
5. Put the Space URL in `web/config.js`.
6. If the Pages address is not `https://samhsoftware.github.io`, change `ALLOWED_ORIGINS`
   in the `Dockerfile`.

A free Space sleeps after a period with no traffic. The page has a "Start model on server"
button which wakes it and waits.
