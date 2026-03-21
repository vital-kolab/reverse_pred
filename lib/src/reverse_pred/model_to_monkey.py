import os
import sys
import numpy as np
from . import prediction_utils as pu

def compute_model_to_monkey(model_features, rates, out_dir, max_n=None, reps=20, out_name='forward_ev', model_type='ridge', save_preds=False):
    os.makedirs(out_dir, exist_ok=True)

    responses = np.nanmean(rates, axis=2)

    ev_path = os.path.join(out_dir, f'{out_name}.npy')

    all_evs = []
    all_preds = []
    for r in range(reps):
        if max_n is not None and model_features.shape[1] > max_n:
            indices = np.random.choice(model_features.shape[1], max_n, replace=False)
            model_features = model_features[:, indices]
        # Compute predictions from model
        prediction = pu.get_all_preds(responses, model_features, ncomp=20, model_type=model_type)

        # Compute EV
        ev = pu.get_all_stats(prediction, rates, model_features, ncomp=20, model_type=model_type)
        all_evs.append(ev)
        all_preds.append(prediction)

    all_evs = np.array(all_evs)
    np.save(ev_path, np.nanmean(all_evs, axis=0))

    if save_preds:
        np.save(os.path.join(out_dir, f'{out_name}_preds.npy'), np.array(all_preds))

