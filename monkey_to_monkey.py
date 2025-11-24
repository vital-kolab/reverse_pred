import os
import sys
import numpy as np
from h5_utils import h5read
import prediction_utils as pu
import h5py

def main(start, end, out_dir, data_dir, reps=10, max_n=None):
    os.makedirs(out_dir, exist_ok=True)

    rates_predictor = np.load("./temp/predictor.npy")
    rates_predicted = np.load("./temp/predicted.npy")
    responses_predicted = np.nanmean(rates_predicted, axis=2)
    responses_predictor = np.nanmean(rates_predictor, axis=2)

    ev_path = os.path.join(out_dir, f'{monkey1}_to_{monkey2}_ev.npy')

    all_evs = []
    for r in range(reps):
        if max_n is not None and responses_predicted.shape[1] > max_n:
            indices = np.random.choice(responses_predicted.shape[1], max_n, replace=False)
            responses_predicted = responses_predicted[:, indices]
            rates_predicted = rates_predicted[:, indices]
        if max_n is not None and responses_predictor.shape[1] > max_n:
            indices = np.random.choice(responses_predictor.shape[1], max_n, replace=False)
            responses_predictor = responses_predictor[:, indices]
            rates_predictor = rates_predictor[:, indices]
        print(responses_predicted.shape, responses_predictor.shape)
        # Compute predictions from model
        prediction = pu.get_all_preds(responses_predicted, responses_predictor, ncomp=20)
        # Compute EV
        ev = pu.get_all_stats(prediction, rates_predicted, rates_predictor, ncomp=20) #, rhoxx, rhoyy
        all_evs.append(ev)

    all_evs = np.array(all_evs)
    np.save(ev_path, np.nanmean(all_evs, axis=0))

if __name__ == "__main__":

    monkey1 = sys.argv[1]
    monkey2 = sys.argv[2]
    out_dir = f'/scratch/smuzelle/results_predictions/monkey2model'
    data_dir = f'./'
    main(start, end, out_dir, data_dir)
