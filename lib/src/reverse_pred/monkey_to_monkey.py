import os
import sys
import numpy as np
from . import prediction_utils as pu

def compute_monkey_to_monkey(rates_predictor, rates_predicted, out_dir, reps=20, max_n=None, name_predicted='monkey2', name_predictor='monkey1'):
    os.makedirs(out_dir, exist_ok=True)

    responses_predicted = np.nanmean(rates_predicted, axis=2)
    responses_predictor = np.nanmean(rates_predictor, axis=2)

    ev_path = os.path.join(out_dir, f'{name_predictor}_to_{name_predicted}_ev.npy')

    all_evs = []
    for r in range(reps):
        if max_n is not None and responses_predictor.shape[1] > max_n:
            indices = np.random.choice(responses_predictor.shape[1], max_n, replace=False)
            responses_predictor = responses_predictor[:, indices]
            rates_predictor = rates_predictor[:, indices]
            
        # Compute predictions from model
        prediction = pu.get_all_preds(responses_predicted, responses_predictor, ncomp=20)
        # Compute EV
        ev = pu.get_all_stats(prediction, rates_predicted, rates_predictor, ncomp=20) #, rhoxx, rhoyy
        all_evs.append(ev)

    all_evs = np.array(all_evs)
    np.save(ev_path, np.nanmean(all_evs, axis=0))
