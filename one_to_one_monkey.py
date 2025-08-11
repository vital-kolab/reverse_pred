import os
import sys
import numpy as np
from h5_utils import h5read
import prediction_utils
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from correlation_metrics import get_splithalf_corr, spearmanbrown_correction

def zscore(train, test):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std


def load_selected_rates(monkey):
    if monkey == "magneto":
        data = h5read('./neural_data/rates_magneto_active.h5', '/magneto/active')
        selected = np.load('./neural_data/selected_rates_magneto.npy')
    elif monkey == "nano":
        data = h5read('./neural_data/rates_nano_active.h5', '/nano/active')
        selected = np.load('./neural_data/selected_rates_nano.npy')
    else:
        raise ValueError("Monkey not found")
    return prediction_utils.average_data(data), selected


def run_neuron_to_neuron(source_rates, target_rates, source_neuron_idx, out_dir, pair_name):
    ev_path = os.path.join(out_dir, f'ev_{pair_name}_neuron{source_neuron_idx}.npy')
    best_target_path = os.path.join(out_dir, f'best_target_{pair_name}_neuron{source_neuron_idx}.npy')

    if os.path.exists(ev_path):
        print(f"Skipping neuron {source_neuron_idx} (already computed)")
        return

    source_responses = np.mean(source_rates, axis=2)[:, source_neuron_idx]
    target_responses = np.mean(target_rates, axis=2)  # shape: (images, target_neurons)

    kf = KFold(n_splits=10, shuffle=True, random_state=11)
    ev_percent = []
    best_targets = []

    for train_idx, test_idx in kf.split(target_responses):
        X_train, X_test = target_responses[train_idx], target_responses[test_idx]
        y_train, y_test = source_responses[train_idx], source_responses[test_idx]

        X_train, X_test = zscore(X_train, X_test)
        y_train, y_test = zscore(y_train[:, None], y_test[:, None])
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        corrs = [pearsonr(X_train[:, n], y_train)[0] for n in range(X_train.shape[1])]
        corrs = np.array(corrs)
        if np.all(np.isnan(corrs)):
            print(f"Neuron {source_neuron_idx}: all correlations are NaN. Skipping.")
            return

        best_target = np.nanargmax(corrs)
        best_targets.append(best_target)

        test_corr = pearsonr(X_test[:, best_target], y_test)[0]

        # compute reliability
        neural_shc_source_all = []
        neural_shc_target_all = []

        X_test_with_trials = target_rates[test_idx, best_target]
        y_test_with_trials = source_rates[test_idx, source_neuron_idx]
        for r in range(10):
            shc_source = get_splithalf_corr(X_test_with_trials,ax=1)
            shc_target = get_splithalf_corr(y_test_with_trials, ax=1)
            neural_shc_source = spearmanbrown_correction(shc_source['split_half_corr'])
            neural_shc_target = spearmanbrown_correction(shc_target['split_half_corr'])
            neural_shc_source_all.append(neural_shc_source)
            neural_shc_target_all.append(neural_shc_target)
        
        neural_shc_source = np.nanmean(neural_shc_source_all)
        neural_shc_target = np.nanmean(neural_shc_target_all)

        ev = test_corr ** 2 / (neural_shc_source * neural_shc_target)
        ev_percent.append(ev * 100)

    np.save(ev_path, np.array(ev_percent))
    np.save(best_target_path, np.array(best_targets))
    print(f"Saved neuron {source_neuron_idx} (mean %EV: {np.nanmean(ev_percent):.2f}%)")


def main(source_monkey, target_monkey, start_idx, group_size):
    out_dir = f'/home/smuzelle/scratch/results_one_to_one/monkey2monkey/{source_monkey}_to_{target_monkey}'
    os.makedirs(out_dir, exist_ok=True)

    _, source_rates = load_selected_rates(source_monkey)
    _, target_rates = load_selected_rates(target_monkey)

    num_neurons = source_rates.shape[1]
    end_idx = min(start_idx + group_size, num_neurons)
    pair_name = f'{source_monkey}_to_{target_monkey}'

    for neuron_idx in range(start_idx, end_idx):
        run_neuron_to_neuron(source_rates, target_rates, neuron_idx, out_dir, pair_name)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python monkey_to_monkey_parallel.py <source_monkey> <target_monkey> <start_index> <group_size>")
        sys.exit(1)

    source_monkey = sys.argv[1]
    target_monkey = sys.argv[2]
    start_idx = int(sys.argv[3])
    group_size = int(sys.argv[4])
    main(source_monkey, target_monkey, start_idx, group_size)
