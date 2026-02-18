import os
import sys
import numpy as np
from . import prediction_utils as pu

def compute_model_to_monkey(model_features, rates, out_dir, reps=20, out_name='forward_ev'):
    os.makedirs(out_dir, exist_ok=True)

    responses = np.nanmean(rates, axis=2)

    ev_path = os.path.join(out_dir, f'{out_name}.npy')

    all_evs = []
    for r in range(reps):
        # Compute predictions from model
        prediction = pu.get_all_preds(responses, model_features, ncomp=20)
        # Compute EV
        ev = pu.get_all_stats(prediction, rates, model_features, ncomp=20)
        all_evs.append(ev)

    all_evs = np.array(all_evs)
    np.save(ev_path, np.nanmean(all_evs, axis=0))

