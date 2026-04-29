# Reverse Predictivity

A research codebase accompanying the paper:

**Reverse predictivity for bidirectional comparison of neural networks and biological brains**, Muzellec & Kar, Nature Machine Intelligence ([nature.com](https://www.nature.com/articles/s42256-026-01204-0))

This repository supports analyses comparing macaque inferior temporal (IT) cortex responses with artificial neural network (ANN) units—specifically using a *reverse predictivity* metric that assesses how well neural responses predict ANN activations.

### Compare brains and models in both directions.

This repository implements **reverse predictivity**: a complementary evaluation to forward neural predictivity that asks *how well do neural responses predict ANN activations?* It provides utilities to map macaque IT population responses to model units, quantify bidirectional alignment, and reproduce manuscript figures.

## 🧠 What is reverse predictivity?
Traditional *forward* neural predictivity evaluates how well a model’s features linearly predict neural responses. **Reverse predictivity** inverts that lens: using neural responses to predict model units. Agreement across both directions strengthens claims that a model and a brain area **share representations**. Practically, this repo includes:

- Regression utilities to decode IT neurons / ANN unit activations from ANN units / IT population responses
- Image‑level metrics and correlation suites to compare human/ANN/neural behaviors
- End‑to‑end notebooks to reproduce figures

## 🗂️ Repository layout

- `demo_forward_predictivity.ipynb` – quick demo of *forward* mapping model units -> neurons
- `demo_reverse_predictivity.ipynb` – quick demo of *reverse* mapping model units <- neurons
- `demo_generate_neurons_i1.ipynb` – compute image‑level neural metrics
- `demo_generate_model_i1.ipynb` – compute image‑level model metrics
- `figure[1-6].ipynb` – figure reproduction notebooks
- `model_to_monkey.py` – utilities for model -> neural regression and evaluation
- `monkey_to_model.py` – utilities for model <- neural regression and evaluation
- `correlation_metrics.py` – Spearman/Pearson, reliability‑aware correlations, confidence intervals
- `regression_metrics.py` – regression helpers
- `prediction_utils.py` – shared helpers for prediction/decoding
- `decode_utils.py` – train/test splits, cross‑validation, split‑half routines
- `figure_utils.py` – journal‑style plotting helpers
- `h5_utils.py` – helpers to read/write HDF5 feature and metadata files

📦 *Large data files (IT features, image sets) are not stored in the repo.* They can be downloaded from: [here](https://osf.io/y3qmk/?view_only=6dfe548c7ba24238932d247e65523053)

## 🛠️ Installation

We recommend Python ≥3.10 with a fresh environment (Conda or venv).

```bash
# Using conda
conda create -n reverse_pred python=3.10 -y
conda activate reverse_pred

# Install core dependencies
pip install numpy scipy scikit-learn matplotlib h5py
```

## 📥 Data & preparation

This project assumes access to:

1. **Macaque IT responses**: population responses for N images.
   - `/neural_data` shape `(n_images, n_neurons, n_reps)`
2. **Model features**: precomputed ANN activations for the same images
   - `/model_features` shape `(n_images, n_units)` 
3. **Humans / Primates behavior**: image‑level accuracies
   - `/behavior` shape `(n_images)`

## 🚀 Quickstart
- `demo_forward_predictivity.ipynb` – step‑by‑step guide to fitting a model to neuron regression, evaluating correlations.
- `demo_reverse_predictivity.ipynb` – end‑to‑end demonstration of neuron to model regression, computing EV/correlation metrics.
- `demo_generate_neurons_i1.ipynb` – generates image‑level accuracies from neural decoders.
- `demo_generate_model_i1.ipynb` – extracts image‑level model metrics from ANN activations.

## 🔁 Reproducing manuscript figures
Each `figureX.ipynb` notebook reproduces the corresponding figure from the preprint. Notebooks expect the data assets described above. If paths differ, change the config cell at the top of each notebook.

- **Figure 1:** Forward Predictivity
- **Figure 2:** Reverse vs forward predictivity examples
- **Figure 3:** Reverse vs forward predictivity accross monkeys and models
- **Figure 4:** Influencing factors
- **Figure 5:** Analysis of unique units
- **Figure 6:** Link with behavior

## 📌 Status & citation
This codebase accompanies the paper:

**Muzellec, Sabine, and Kohitij Kar. "Reverse predictivity for bidirectional comparison of neural networks and biological brains." Nature Machine Intelligence 8.3 (2026): 474-488.**

If you use this repository or ideas from it, please cite the paper and link to this repo.

```
@article{muzellec2026reverse,
  title={Reverse predictivity for bidirectional comparison of neural networks and biological brains},
  author={Muzellec, Sabine and Kar, Kohitij},
  journal={Nature Machine Intelligence},
  volume={8},
  number={3},
  pages={474--488},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

License: **MIT** (see `LICENSE`).
