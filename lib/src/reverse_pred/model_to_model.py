import os
import sys
import numpy as np
from . import prediction_utils as pu

def compute_model_to_model(model_features_predictor, model_features_predicted, out_dir, reps=20, name_predicted='model2', name_predictor='model1', model_type='ridge', save_preds=False): 
    os.makedirs(out_dir, exist_ok=True)
    ev_path = os.path.join(out_dir, f'{name_predictor}_to_{name_predicted}_ev.npy')

    all_evs = []
    all_preds = []
    for r in range(reps):
        # Compute predictions from model
        prediction = pu.get_all_preds(model_features_predicted, model_features_predictor, ncomp=20, model_type=model_type)
        # Compute EV
        ev = pu.get_all_stats(prediction, model_features_predicted, model_features_predictor, ncomp=20, model_type=model_type)
        all_evs.append(ev)
        all_preds.append(prediction)
    
    all_evs = np.array(all_evs)
    np.save(ev_path, np.nanmean(all_evs, axis=0))

    if save_preds:
        np.save(os.path.join(out_dir, f'{name_predictor}_to_{name_predicted}_preds.npy'), np.array(all_preds))

