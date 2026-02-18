import os
import sys
import numpy as np
from . import prediction_utils as pu

def compute_monkey_to_model(model_features, rates, out_dir, max_n=None, reps=20, out_name='reverse_ev', model_type='ridge'):
    os.makedirs(out_dir, exist_ok=True)

    responses = np.nanmean(rates, axis=2)

    ev_path = os.path.join(out_dir, f'{out_name}.npy')

    all_evs = []
    for r in range(reps):
        if max_n is not None and responses.shape[1] > max_n:
            indices = np.random.choice(responses.shape[1], max_n, replace=False)
            responses = responses[:, indices]
            rates = rates[:, indices]
        # Compute predictions from model
        prediction = pu.get_all_preds(model_features, responses, ncomp=20, model_type=model_type)
        # Compute EV
        ev = pu.get_all_stats(prediction, model_features, rates, ncomp=20, model_type=model_type)
        all_evs.append(ev)

    all_evs = np.array(all_evs)
    np.save(ev_path, np.nanmean(all_evs, axis=0))
