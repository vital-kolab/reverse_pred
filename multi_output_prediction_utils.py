from scipy import stats
from regression_metrics import pls_regress, lasso_regress, get_train_test_indices, mlp_regress, ridge_regress
import numpy as np
from correlation_metrics import get_splithalves, spearmanbrown_correction

def get_predictions_multioutput(responses, predictor, ncomp=10, nrfolds=10, seed=0, model=None, monkey=None):
    nrImages, n_targets = responses.shape
    ypred = np.full((nrImages, n_targets), np.nan)

    for i in range(nrfolds):
        train, test = get_train_test_indices(nrImages, nrfolds=nrfolds, foldnumber=i, seed=seed)
        #pred = pls_regress(predictor[train, :], responses[train, :], predictor[test, :], ncomp=ncomp, model=model, monkey=monkey, fold=i)
        pred = ridge_regress(predictor[train, :], responses[train, :], predictor[test, :], model=model, monkey=monkey, fold=i)
        ypred[test, :] = pred

    return ypred

# Updated to handle multi-target
def get_all_preds(neurons_predicted, neurons_predictor, ncomp, model=None, monkey=None):
    if len(neurons_predicted.shape) == 3:
        mean_target = np.mean(neurons_predicted, axis=2)   # shape: (n_images, n_target_neurons)
    else:
        mean_target = neurons_predicted

    if len(neurons_predictor.shape) == 3:
        mean_source = np.mean(neurons_predictor, axis=2)   # shape: (n_images, n_source_neurons)
    else:
        mean_source = neurons_predictor
    p = get_predictions_multioutput(mean_target, mean_source, ncomp=ncomp, model=model, monkey=monkey)
    return p

def get_splithalf_corr(var, ax=1, type='spearman'):
    _, _, split_mean1, split_mean2 = get_splithalves(var, ax=ax)  # e.g., output shape (samples, neurons)
    
    # Make sure the inputs are 2D
    assert split_mean1.ndim == 2 and split_mean2.ndim == 2, "Split halves must be 2D"

    correlations = []
    for i in range(split_mean1.shape[1]):  # iterate over neurons
        x, y = split_mean1[:, i], split_mean2[:, i]
        if type == 'spearman':
            r, _ = stats.spearmanr(x, y)
        else:
            r, _ = stats.pearsonr(x, y)
        correlations.append(r)

    return {
        'split_half_corr': np.array(correlations),
        'type': type
    }

def predictivity(x, y, rho_xx, rho_yy):
    assert x.shape == y.shape, "Input and prediction shapes must match"
    n_neurons = x.shape[1]

    raw_corr = np.array([stats.pearsonr(x[:, i], y[:, i])[0] for i in range(n_neurons)])
    denominator = np.sqrt(rho_xx * rho_yy)
    corrected_raw_corr = raw_corr / denominator
    ev = (corrected_raw_corr ** 2) * 100
    return ev, raw_corr, corrected_raw_corr


def get_neural_neural_splithalfcorr(rate_predicted, rate_predictor, ncomp=10, nrfolds=10, seed=0):
    # Split-half correlation of each predicted neuron
    print(rate_predicted.shape)
    shc_predicted = get_splithalf_corr(rate_predicted, ax=2)  # shape: (n_neurons,) or (n_neurons, n_neurons)
    print(shc_predicted['split_half_corr'].shape)
    # Predict using split 1 and split 2 of the predictor
    sp1_predictor, sp2_predictor, _, _ = get_splithalves(rate_predictor, ax=2)

    p1 = get_predictions_multioutput(np.mean(rate_predicted, axis=2), np.mean(sp1_predictor, axis=2),
                                     nrfolds=nrfolds, ncomp=ncomp, seed=seed)
    p2 = get_predictions_multioutput(np.mean(rate_predicted, axis=2), np.mean(sp2_predictor, axis=2),
                                     nrfolds=nrfolds, ncomp=ncomp, seed=seed)

    prediction_shc = np.array([stats.pearsonr(p1[:, i], p2[:, i])[0] for i in range(p1.shape[1])])
    prediction_shc = spearmanbrown_correction(prediction_shc)

    # Fix here: extract diagonals from full matrix
    if isinstance(shc_predicted, dict):
        mat = shc_predicted['split_half_corr']
    else:
        mat = shc_predicted

    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        diag_vals = np.diag(mat)
    else:
        diag_vals = mat

    neuron_shc = spearmanbrown_correction(diag_vals)

    return prediction_shc, neuron_shc

def get_neural_model_splithalfcorr(model_features, rate, ncomp=10, nrfolds=10, seed=0):
    """
    model_features: shape (n_images, n_model_units) - deterministic
    rate: shape (n_images, n_neurons, n_repeats) - noisy
    """
    sp1, sp2, _, _ = get_splithalves(rate, ax=2)  # split neural responses along repetitions

    # Predict each split of neural data from fixed model features
    p1 = get_predictions_multioutput(model_features, np.mean(sp1, axis=2), nrfolds=nrfolds, ncomp=ncomp, seed=seed)
    p2 = get_predictions_multioutput(model_features, np.mean(sp2, axis=2), nrfolds=nrfolds, ncomp=ncomp, seed=seed)

    # Compute split-half correlation per neuron
    corr = np.array([stats.pearsonr(p1[:, i], p2[:, i])[0] for i in range(p1.shape[1])])
    model_shc = spearmanbrown_correction(corr)

    return model_shc, 1.0

def get_model_neural_splithalfcorr(rate, model_features, ncomp=10, nrfolds=10, seed=0):
    """
    Predict noisy neural responses from model features.
    - rate: shape (images, neurons, repeats)
    - model_features: shape (images, model_units)
    """
    # Split the rate data along the repetition axis
    sp1, sp2, _, _ = get_splithalves(rate, ax=2)

    # Compute SHC for the neural rate
    shc = get_splithalf_corr(rate, ax=2)

    # Model predictions from averaged neural splits
    target_sp1 = np.mean(sp1, axis=2)  # (images, neurons)
    target_sp2 = np.mean(sp2, axis=2)  # (images, neurons)

    # Predict both splits from the model features
    p1 = get_predictions_multioutput(target_sp1, model_features, nrfolds=nrfolds, ncomp=ncomp, seed=seed)
    p2 = get_predictions_multioutput(target_sp2, model_features, nrfolds=nrfolds, ncomp=ncomp, seed=seed)

    # Compute split-half correlation of model predictions per neuron
    model_shc = np.array([stats.pearsonr(p1[:, i], p2[:, i])[0] for i in range(p1.shape[1])])
    model_shc = spearmanbrown_correction(model_shc)

    neural_shc = spearmanbrown_correction(shc['split_half_corr'])

    return model_shc, neural_shc

def get_all_stats(p, neurons_predicted, neurons_predictor, ncomp):
    if len(neurons_predicted.shape) == 3:
        mean_target = np.mean(neurons_predicted, axis=2)   # shape: (n_images, n_target_neurons)
    else:
        mean_target = neurons_predicted

    if len(neurons_predicted.shape) == 3 and len(neurons_predictor.shape) == 3:
        mshc, nshc = get_neural_neural_splithalfcorr(neurons_predicted, neurons_predictor, ncomp=ncomp)

    if len(neurons_predicted.shape) == 2 and len(neurons_predictor.shape) == 3:
        mshc, nshc = get_neural_model_splithalfcorr(neurons_predicted, neurons_predictor, ncomp=ncomp)

    if len(neurons_predicted.shape) == 3 and len(neurons_predictor.shape) == 2:
        mshc, nshc = get_model_neural_splithalfcorr(neurons_predicted, neurons_predictor, ncomp=ncomp)

    if len(neurons_predicted.shape) == 2 and len(neurons_predictor.shape) == 2:
        mshc, nshc = 1.0, 1.0

    ev, _, _ = predictivity(mean_target, p, nshc, mshc)  # Now p and mean_target are 2D
    return ev  # shape: (n_target_neurons,)