import os
import sys
import numpy as np
from h5_utils import h5read
import prediction_utils
import multi_output_prediction_utils as mu
import h5py

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


def load_model_features(model):
    path_map = {
        #"alexnet": "model_features/alexnet_features.npy",
        "alexnet_relu": "model_features/alexnet_features_layer_last.npy",
        "resnet": "model_features/resnet_features.npy",
        "resnet18": "model_features/resnet18_features.npy",
        "resnet18_ssl": "model_features/resnet18_ssl_features.npy",
        "resnet18_robust": "model_features/resnet18_robust_features.npy",
        "resnet_ssl": "model_features/resnet50_ssl_features.npy",
        "resnet_swsl": "model_features/resnet50_swsl_features.npy",
        "resnetSIN": "model_features/resnetSIN_features.npy",
        #"resnet152": "model_features/resnet152_features.npy",
        "resnet152_2": "model_features/resnet152_2_features.npy",
        "resnet101": "model_features/resnet101_features.npy",
        "resnet101_ssl": "model_features/resnet101_ssl_features.npy",
        "resnet_robust_eps1": "model_features/resnet_robust_eps1_features.npy",
        "resnet_robust_eps3": "model_features/resnet_robust_eps3_features.npy",
        "inception": "model_features/inceptionv3_features.npy",
        "inceptionv1": "model_features/inceptionv1_features.npy",
        "vit": "model_features/vit_features.npy",
        "vit_ssl": "model_features/vit_ssl_features.npy",
        "vgg16": "model_features/vgg16_features.npy",
        "vgg16_robust": "model_features/vgg16_robust_eps3_features.npy",
        "densenet": "model_features/densenet_features.npy",
        "densenet169": "model_features/densenet169_features.npy",
        "densenet161": "model_features/densenet161_features.npy",
        "densenet161_robust": "model_features/densenet161_robust_eps3_features.npy",
        "densenet121": "model_features/densenet121_features.npy",
        "convnext": "model_features/convnext_features.npy",
        "convnext_ssl": "model_features/convnext_ssl_features.npy",
        "mobilenet": "model_features/mobilenet_features.npy",
        "mobilenet_v2": "model_features/mobilenet_v2_features.npy",
        "mobilenet_robust": "model_features/mobilenet_robust_eps3_features.npy",
        "squeezenet": "model_features/squeezenet_features.npy",
        #"cornetS": "model_features/cornetS_features.npy",
        "cornetS_t0": "model_features/CORnet-S_muri1320_IT_output_feats_ts_0.npy",
        "cornetS_reg": "model_features/cornetS_regularized_features.npy",
        "cornetS_reg2": "model_features/cornetS_regularized_v2_features.npy",
        "cornetRT_t4": "model_features/CORnet-RT_muri1320_IT_output_feats_ts_4.npy",
        "nasnet": "model_features/nasnet_features.npy",
        "pnasnet": "model_features/pnasnet_features.npy",
        "swin": "model_features/swin_features.npy",
        "swin_ssl": "model_features/swin_ssl_features.npy",
        "shufflenet": "model_features/shufflenet_features.npy",
        "shufflenet_robust": "model_features/shufflenet_robust_features.npy",
    }
    if model not in path_map:
        raise ValueError("Model not found")
    features = np.load(path_map[model])
    return features.reshape((1320, -1))

def load_predictor_model_features(model):
    path_map = {
        "resnet_0": "model_features/muri1320_resnet50_diffInitRes0_IT.h5",
        "resnet_1": "model_features/muri1320_resnet50_diffInitRes1_IT.h5",
        "resnet_2": "model_features/muri1320_resnet50_diffInitRes2_IT.h5",
        "resnet_3": "model_features/muri1320_resnet50_diffInitRes3_IT.h5",
        "resnet_4": "model_features/muri1320_resnet50_diffInitRes4_IT.h5",
        "resnet_5": "model_features/muri1320_resnet50_diffInitRes5_IT.h5",
        "resnet_6": "model_features/muri1320_resnet50_diffInitRes6_IT.h5",
        "resnet_7": "model_features/muri1320_resnet50_diffInitRes7_IT.h5",
        "resnet_8": "model_features/muri1320_resnet50_diffInitRes8_IT.h5",
        "resnet_9": "model_features/muri1320_resnet50_diffInitRes9_IT.h5",
        "resnet_10": "model_features/muri1320_resnet50_diffInitRes10_IT.h5",
        "resnet_11": "model_features/muri1320_resnet50_diffInitRes11_IT.h5",
        "resnet_12": "model_features/muri1320_resnet50_diffInitRes12_IT.h5",
}

    with h5py.File(path_map[model], 'r') as f:
        data_name = list(f.keys())[0]  # Replace with actual dataset name if known
        data = f[data_name][:]

    return data

def predict_all(model_features, worst_neurons):
    #model_features.remove(neuron_index)
    
        
    prediction = mu.get_all_preds(worst_neurons, model_features, ncomp=20)
    #np.save(pred_path, prediction)

    ev = mu.get_all_stats(prediction, worst_neurons, model_features, ncomp=20)
    #np.save(ev_path, ev)

    return ev

def main(model1, model2, n_neurons):
    out_dir = f'./results_predictions/within_model_v4/{model1}/'
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed=n_neurons)

    model_features = load_model_features(model1)
    predictor_features = load_predictor_model_features(model2)
    ev = np.load(f'./results_predictions/monkey2model/magneto/v4/ev_{model1}.npy')

    ev_path = os.path.join(out_dir, f'ev_{model2}_{n_neurons}.npy')

    worst_neurons = model_features[:, ev < np.nanpercentile(ev, 20, axis=0)]
    worst_neurons = worst_neurons[:,:500]

    worst_indices = np.where(ev < np.nanpercentile(ev, 20))[0][:500]
    model_features = np.delete(model_features, worst_indices, axis=1)

    ev_sampling = []
    for r in range(10):
        sampled_indices = rng.choice(predictor_features.shape[1], size=n_neurons, replace=False)
        predictor_features_sampled = predictor_features[:, sampled_indices]
        ev = predict_all(predictor_features_sampled, worst_neurons)
        ev_sampling.append(ev)

    np.save(ev_path, ev_sampling)

if __name__ == "__main__":
    # if len(sys.argv) != 6:
    #     print("Usage: python within_model_predict_chunked.py <model> <monkey> <start_index> <chunk_size> <n_sampled_units> <n_repeats>")
    #     sys.exit(1)

    sizes = np.linspace(20, 100352, 20).astype(int)
    for n_neurons in sizes:
        print(n_neurons)
        model1 = sys.argv[1]
        model2 = sys.argv[2]
        main(model1, model2, n_neurons)
