import os
import sys
import numpy as np
import h5py
from h5_utils import h5read
import prediction_utils as pu

def load_model_features(model_name, n_images, data_dir):
    features = np.load(os.path.join(data_dir, f"{model_name}_features.npy")).reshape(n_images, -1)
    return features

def main(model1, model2, out_dir, n_images, data_dir, reps=10):
    os.makedirs(out_dir, exist_ok=True)
    ev_path = os.path.join(out_dir, f'forward_{model2}_ev.npy')
    if not os.path.exists(ev_path):
        model_features_predictor = load_model_features(model1, n_images, data_dir)
        model_features_predicted = load_model_features(model2, n_images, data_dir)

        # Compute predictions from model
        prediction = pu.get_all_preds(model_features_predicted, model_features_predictor, ncomp=20)
        # Compute EV
        ev = pu.get_all_stats(prediction, model_features_predicted, model_features_predictor, ncomp=20)
        print(np.nanmean(ev))
        np.save(ev_path, ev)

if __name__ == "__main__":

    model1 = sys.argv[1]
    model2 = sys.argv[2]
    out_dir = f'/scratch/smuzelle/results_predictions/model2model/{model1}'
    data_dir = f'/scratch/smuzelle/model_features/'
    n_images = 1320
    main(model1, model2, out_dir, n_images, data_dir)
