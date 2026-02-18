# Reverse Predictivity

A lightweight, modular Python library for computing **bidirectional alignment** between artificial neural network (ANN) representations and primate inferior temporal (IT) cortex responses.

This package accompanies the preprint:

**Muzellec & Kar (2025). _Reverse Predictivity: Going Beyond One-Way Mapping to Compare Artificial Neural Network Models and Brains_. bioRxiv.**
<https://www.biorxiv.org/content/10.1101/2025.08.08.669382v1>

Reverse predictivity complements traditional *forward neural predictivity* by asking the reciprocal question:

> **How well do neural responses predict ANN activations?**

Together, the forward and reverse metrics provide a more complete picture of representational similarity between brains and models.

---

## 🧠 Library Overview

This library contains four core mapping modules:

| Module | Mapping Direction | Question Answered |
|--------|-------------------|-------------------|
| `model_to_monkey.py` | Model → Monkey | How well do ANN features predict neural responses? *(forward predictivity)* |
| `monkey_to_model.py` | Monkey → Model | How well do IT neurons predict ANN unit activations? *(reverse predictivity)* |
| `monkey_to_monkey.py` | Monkey A → Monkey B | How consistent are neural populations across animals? *(biological upper bound)* |
| `model_to_model.py` | Model A → Model B | How aligned are representations across models or layers? |

All functions compute **explained variance (EV)** using repeated linear mappings and save EV arrays to disk.

---

## 🔧 Installation

We recommend using a clean environment:

```bash
conda create -n reverse_pred python=3.10 -y
conda activate reverse_pred
```

Install required Python packages:

```bash
pip install numpy scipy scikit-learn matplotlib
```

Install this library:

```bash
pip install reverse_pred
```

---

## 🚀 Usage

Each mapping function takes:

- `model_features`: `(n_images × n_units)` array
- `rates`: `(n_images × n_neurons × n_repeats)` array
- `out_dir`: output directory for saving EV results
- `reps`: number of cross-validated fits (default: 20)
- `model_type`: Choice of regressor models among `ridge, linear, lasso, elasticnet, pls`. Default=`ridge`

---

### 1. Forward Predictivity

**Module:** `model_to_monkey.py`  
**Function:** `compute_model_to_monkey`

```python
from reverse_predictivity.model_to_monkey import compute_model_to_monkey
import numpy as np

model_features = np.load("features/resnet50_itlayer.npy")
rates = np.load("data/it_rates.npy")

compute_model_to_monkey(
    model_features=model_features,
    rates=rates,
    out_dir="results/model_to_monkey/resnet50",
    reps=20,
    out_name='forward_ev'
)
```

**Output:**

```
results/model_to_monkey/resnet50/forward_ev.npy
```

---

### 2. Reverse Predictivity

**Module:** `monkey_to_model.py`  
**Function:** `compute_monkey_to_model`

```python
from reverse_predictivity.monkey_to_model import compute_monkey_to_model
import numpy as np

model_features = np.load("features/resnet50_itlayer.npy")
rates = np.load("data/it_rates.npy")

compute_monkey_to_model(
    model_features=model_features,
    rates=rates,
    out_dir="results/monkey_to_model/resnet50",
    max_n=None,
    reps=20,
    out_name='reverse_ev'
)
```

**Parameters:**

`max_n`: can be set to subsample n number of neurons within the neural population. Each repetition will be done using a different sampling.

**Output:**

```
results/monkey_to_model/resnet50/reverse_ev.npy
```

---

### 3. Neural–Neural Consistency

**Module:** `monkey_to_monkey.py`  
**Function:** `compute_monkey_to_monkey`

```python
from reverse_predictivity.monkey_to_monkey import compute_monkey_to_monkey
import numpy as np

ratesA = np.load("data/monkeyA_rates.npy")
ratesB = np.load("data/monkeyB_rates.npy")

compute_monkey_to_monkey(
    rates_predictor=ratesA,
    rates_predicted=ratesB,
    out_dir="results/monkey_to_monkey/",
    reps=20,
    max_n=None,
    name_predicted="monkeyB",
    name_predictor="monkeyA",
)
```

**Parameters:**

`max_n`: can be set to subsample n number of predictor neurons. Each repetition will be done using a different sampling.

**Output:**

```
results/monkey_to_monkey/monkeyA_to_monkeyB_ev.npy
```

---

### 4. Model–Model Alignment

**Module:** `model_to_model.py`  
**Function:** `compute_model_to_model`

```python
from reverse_predictivity.model_to_model import compute_model_to_model
import numpy as np

modelA = np.load("features/resnet50_itlayer.npy")
modelB = np.load("features/convnext_itlayer.npy")

compute_model_to_model(
    model_features_predictor=modelA,
    model_features_predicted=modelB,
    out_dir="results/model_to_model/resnet_to_convnext",
    reps=20,
    name_predicted="convnext",
    name_predictor="resnet50",
)
```

**Output:**

```
results/model_to_model/resnet_to_convnext/resnet50_to_convnext_ev.npy
```

---

## 📊 Downstream Analysis

After generating EV results:

1. Load the saved `.npy` files.
2. Compare forward vs reverse predictivity.
3. Compare model–monkey EV to monkey–monkey EV.
4. Compare model–model EV across architectures.

```python
import numpy as np
import matplotlib.pyplot as plt

fwd = np.load("results/model_to_monkey/resnet50/forward_ev.npy")
rev = np.load("results/monkey_to_model/resnet50/reverse_ev.npy")

plt.hist(fwd, bins=30, alpha=0.6, label="Forward")
plt.hist(rev, bins=30, alpha=0.6, label="Reverse")
plt.legend()
plt.xlabel("Explained Variance")
plt.ylabel("Count")
plt.show()
```

---

## 📌 Citation

If you use this library, please cite:

```
@article{muzellec_kar_2025_reversepredictivity,
  title   = {Reverse Predictivity: Going Beyond One-Way Mapping to Compare Artificial Neural Network Models and Brains},
  author  = {Muzellec, Sabine and Kar, Kohitij},
  journal = {bioRxiv},
  year    = {2025}
}
```

---

## 📜 License

MIT License — see `LICENSE`.
