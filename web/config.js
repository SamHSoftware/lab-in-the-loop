// Address of the Python API. Replace YOUR-HF-USERNAME after you create the Space.
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);

window.LAB_CONFIG = {
  apiUrl: IS_LOCAL
    ? "http://localhost:7860"
    : "https://YOUR-HF-USERNAME-lab-in-the-loop.hf.space",
  // Must match N_SCREEN_EXPERIMENTS in lab_in_the_loop/loop.py.
  screenExperiments: 5,
};
