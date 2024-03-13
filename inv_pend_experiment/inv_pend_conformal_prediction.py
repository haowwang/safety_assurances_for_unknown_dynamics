import numpy as np
import torch
import os, sys
from scipy.io import loadmat, savemat
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))
sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )))

from utils import ensemble_forward_initialize


# x_dot_0 = x[:, [1]] # x0 dot = theta dot = x1
# x_dot_1 = (-b * x[:, 1] + (m*g*l*np.sin(x[:, 0]))/2 - u) / ((m*l**2)/3)



def f1_ground_truth_func(state_input, m,l,g,b):
    dim_0 = state_input[:,[1]]
    dim_1 = (-b * state_input[:, [1]] + (m*g*l*np.sin(state_input[:, [0]]))/2) / ((m*l**2)/3)
    return np.hstack((dim_0, dim_1))

def f2_ground_truth_func(state_input, m,l):
    return np.array([[0, -1/((m*l**2)/3)]]).repeat(state_input.shape[0],0)


def main():
    base_path = base_path = os.path.dirname(os.path.realpath(__file__))
    m = 0.5
    l = 1.
    b = 0.1
    g = 9.8
    num_validation_samples = 5000
    STATE_DIM = 2
    CONTROL_DIM = 1

    dyn_model_idx_list = [0,1,2,3,4]

    ensemble_forward = ensemble_forward_initialize(state_dim=STATE_DIM, control_dim=CONTROL_DIM, dyn_model_idx_list=dyn_model_idx_list, 
                                                    dyn_experiment_name='INV_PEND_DATA1H', base_path = base_path, if_gpu=True, 
                                                    batch_size = 1024, model_checkpoint_epoch='final', mode='eval_without_ctrl')


    dyn_dataset = loadmat(os.path.join(base_path, 'dyn_datasets', 'inv_pend_data1h_dyn_dataset.mat'))
    dyn_dataset_summary = loadmat(os.path.join(base_path, 'dyn_datasets', 'inv_pend_data1h_dyn_dataset_summary.mat'))
    dyn_train_labels_mean = dyn_dataset_summary['labels_mean']
    dyn_train_labels_std = dyn_dataset_summary['labels_std']
    eval_inputs = dyn_dataset['eval_inputs']

    rand_sample_idx = np.random.randint(low=0, high = eval_inputs.shape[0], size=(num_validation_samples))
    sampled_input = torch.tensor(eval_inputs[rand_sample_idx, :2], dtype = torch.float32)
    ensemble_dyn_forward_results = ensemble_forward(dyn_forward_samples=sampled_input)
    net_f1_output = ensemble_dyn_forward_results['x_dot_f1_mean'] # shape (num_validation_samples, 2)
    net_f2_output = ensemble_dyn_forward_results['x_dot_f2_mean'].squeeze() # shape (num_validation_samples, 2)
    

    assert net_f1_output.ndim == net_f2_output.ndim == 2
    net_f1_f2_output = np.hstack((net_f1_output, net_f2_output)) # shape (num_validation_samples, 4)

    f1_ground_truth = f1_ground_truth_func(sampled_input.numpy(), m=m, l=l, g=g, b=b)
    f2_ground_truth = f2_ground_truth_func(sampled_input.numpy(), m=m, l=l)
    f1_ground_truth_normalized = (f1_ground_truth - dyn_train_labels_mean) / dyn_train_labels_std
    f2_ground_truth_normalized = f2_ground_truth / dyn_train_labels_std


    f1_f2_ground_truth_normalized = np.hstack((f1_ground_truth_normalized, f2_ground_truth_normalized))
    f1_f2_abs_error = np.abs(f1_f2_ground_truth_normalized - net_f1_f2_output)
    quantile_95th_abs_error = np.quantile(f1_f2_abs_error, 0.95, axis=0, keepdims=True)
    print(quantile_95th_abs_error)

    f1_std_conf_pred = quantile_95th_abs_error[:, :2]
    f2_std_conf_pred = quantile_95th_abs_error[:, 2:, np.newaxis]

    dyn_forward_samples = torch.tensor(loadmat('inv_pend_pi_10_200_200_grid_data.mat')['state_input_to_net'], dtype = torch.float32)
    ensemble_dyn_forward_results = ensemble_forward(dyn_forward_samples=dyn_forward_samples)
    ensemble_dyn_forward_results['x_dot_f1_std'] = np.repeat(f1_std_conf_pred, dyn_forward_samples.shape[0], axis = 0)
    ensemble_dyn_forward_results['x_dot_f2_std'] = np.repeat(f2_std_conf_pred, dyn_forward_samples.shape[0], axis = 0)

    savemat('ensemble_dyn_forward_results/inv_pend_data1h_conf_pred_ensemble_forward_results.mat', ensemble_dyn_forward_results)


if __name__ == '__main__':
    main()

