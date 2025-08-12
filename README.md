# Reverse Predictivity

A research codebase accompanying the preprint:

**Reverse Predictivity: Going Beyond One-Way Mapping to Compare Artificial Neural Network Models and Brains**, Muzellec & Kar, bioRxiv (posted August 8, 2025) ([biorxiv.org](https://www.biorxiv.org/content/10.1101/2025.08.08.669382v1.full.pdf?utm_source=chatgpt.com))

This repository supports analyses comparing macaque inferior temporal (IT) cortex responses with artificial neural network (ANN) units—specifically using a *reverse predictivity* metric that assesses how well neural responses predict ANN activations ([biorxiv.org](https://www.biorxiv.org/content/10.1101/2025.08.08.669382v1?utm_source=chatgpt.com)).

---

##  Repository Contents

| Component | Purpose |
|---|---|
| `multi_output_prediction_utils.py` | Utilities for multi‑output regression and cross‑validation. |
| `within_model_prediction_v2.py` | Assess recoverability of features *within* an ANN. |
| `between_instances_prediction_v2.py` | Evaluate generalization across model instances or augmentations. |
| `model_to_model_multi_ouput.py` | Forward mapping between model layers or architectures. |
| `model_to_monkey_multi_ouput.py` | Predict neural activity from model features. |
| `monkey_to_model_multi_ouput.py` | **Reverse predictivity**: predict ANN activations from neural responses. |
| `monkey_to_model_unique_multi_ouput.py` | Reverse predictivity focused on unique model units. |
| `one_to_one_monkey.py` | Unit-level mappings between neural and model units. |
| `get_features.ipynb` | Notebook to extract ANN features given images. |
| `plot_reverse_predictivity_v2.ipynb` | Visualization of reverse predictivity results. |
| `plots_behavior_v2.ipynb`, `plot_behavior_corrected_for_accuracy.ipynb` | Behavior-focused plotting, including image-wise signatures and accuracy controls. |
| `LICENSE` | MIT License. |

---

##  Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install numpy scipy scikit-learn pandas matplotlib jupyter
pip install torch torchvision torchaudio  # If using PyTorch feature extraction
```

Or, later:
```bash
pip install -r requirements.txt
```

---

##  Data Requirements

You'll need:
- **ANN features**: `[N_images, N_units]`, per model/layer.
- **Neural responses**: e.g., macaque IT `[N_images, N_neurons]`.
- Optional: **behavioral data** aligned to images.
All must be ordered consistently across images.

---

##  Usage Overview

###  1. Extract Features
Run `get_features.ipynb` to process your image stimuli through selected ANN(s) and save activations.

###  2. Run Mappings
```bash
python monkey_to_model_multi_ouput.py   --neural path/to/neural.npy   --features path/to/model_features.npy   --cv-folds 5   --alpha-grid 0.001,0.01,0.1,1
```
This evaluates how well neural data predicts model activations using ridge regression (aligned with the reverse predictivity concept) ([biorxiv.org](https://www.biorxiv.org/content/10.1101/2025.08.08.669382v1?utm_source=chatgpt.com)).

Similarly:
```bash
python model_to_monkey_multi_ouput.py ...        # Forward mapping
python within_model_prediction_v2.py ...         # Within-model mapping
python between_instances_prediction_v2.py ...    # Cross-instance mapping
```

###  3. Visualization
- `plot_reverse_predictivity_v2.ipynb` for ANN–brain mapping results.
- `plots_behavior_v2.ipynb` and `plot_behavior_corrected_for_accuracy.ipynb` for behavioral analyses.

---

##  Research Context

The **Reverse Predictivity** metric quantifies the degree to which macaque IT neural responses can predict individual ANN unit activations—shifting beyond traditional forward predictivity approaches ([biorxiv.org](https://www.biorxiv.org/content/10.1101/2025.08.08.669382v1?utm_source=chatgpt.com)).

The analyses explore:
- Mapping within/between models.
- Mapping across brains and models.
- Behavioral alignment.

The paper introduces and validates this framework by comparing model activations with primate neural data in a bidirectional mapping context ([biorxiv.org](https://www.biorxiv.org/content/10.1101/2025.08.08.669382v1?utm_source=chatgpt.com)).

---

##  Reproducibility Tips

- Fix random seeds (e.g., `random_state=42`).
- Use stratified 或 `GroupKFold` if relevant.
- Estimate neural signal reliability (e.g. split-half) to contextualize predictivity ceilings.
- Experiment with ridge vs. Lasso regressors to verify robustness.

---

##  Contributing

Contributions are appreciated:

1. Open an issue describing your idea or bug.
2. Implement changes with clear docstrings and type hints.
3. Provide test snippets or notebooks when possible.

---

##  Citation

If using this codebase and/or the associated preprint, please cite:

```
Muzellec, S., & Kar, K. (2025, August 8). Reverse Predictivity: Going Beyond One-Way Mapping to Compare Artificial Neural Network Models and Brains. bioRxiv.
```
