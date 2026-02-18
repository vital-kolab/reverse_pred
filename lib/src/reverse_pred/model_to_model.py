import os
import sys
import numpy as np
from . import prediction_utils as pu

def compute_model_to_model(model_features_predictor, model_features_predicted, out_dir, reps=20, name_predicted='model2', name_predictor='model1'):
    os.makedirs(out_dir, exist_ok=True)
    ev_path = os.path.join(out_dir, f'{name_predictor}_to_{name_predicted}_ev.npy')

    all_evs = []
    for r in range(reps):
        # Compute predictions from model
        prediction = pu.get_all_preds(model_features_predicted, model_features_predictor, ncomp=20)
        # Compute EV
        ev = pu.get_all_stats(prediction, model_features_predicted, model_features_predictor, ncomp=20)
        all_evs.append(ev)
    
    all_evs = np.array(all_evs)
    np.save(ev_path, np.nanmean(all_evs, axis=0))

