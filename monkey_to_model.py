import os
import sys
import numpy as np
from h5_utils import h5read
import prediction_utils as pu
import h5py

def load_selected_rates(monkey, data_dir):
    if monkey == "magneto":
        data = h5read(os.path.join(data_dir, 'neural_data/rates_magneto_active.h5'), '/magneto/active')
        selected = np.load(os.path.join(data_dir, 'neural_data/selected_rates_magneto.npy'))
    elif monkey == "nano":
        data = h5read(os.path.join(data_dir, 'neural_data/rates_nano_active.h5'), '/nano/active')
        selected = np.load(os.path.join(data_dir, 'neural_data/selected_rates_nano.npy'))
    else:
        raise ValueError("Monkey not found")
    return pu.average_data(data), selected


def load_features(model_name, n_images, data_dir):
    features = np.load(os.path.join(data_dir, f'model_features/{model_name}_features.npy')).reshape(n_images, -1)
    return features

def load_model_features(model, n_images, data_dir):
    features = load_features(model, n_images, data_dir)
    return features

def main(model, monkey, out_dir, n_images, data_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Load model features and data
    rates, _ = load_selected_rates(monkey, data_dir)
    model_features = load_model_features(model, n_images, data_dir)
    responses = np.nanmean(rates, axis=2)
    print(responses.shape)

    ev_path = os.path.join(out_dir, f'reverse_{monkey}_ev.npy')
    
    # Compute predictions from model
    prediction = pu.get_all_preds(model_features, responses, ncomp=20)
    # Compute EV
    ev = pu.get_all_stats(prediction, model_features, rates, ncomp=20)
    np.save(ev_path, ev)

if __name__ == "__main__":

    model = sys.argv[1]
    monkey = sys.argv[2]
    out_dir = f'./results_predictions/monkey2model/{model}'
    data_dir = f'./'
    n_images = 1320
    main(model, monkey, out_dir, n_images, data_dir)
