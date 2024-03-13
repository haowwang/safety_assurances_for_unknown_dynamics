import os, sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))
sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )))
from scipy.io import loadmat, savemat
import torch
from utils import ensemble_forward_initialize




if __name__ == "__main__":
    STATE_DIM = 2
    CONTROL_DIM = 1
    dyn_model_idx_list = [0,1,2,3,4]
    model_checkpoint_epoch = 'final'
    dyn_forward_samples = torch.tensor(loadmat('inv_pend_pi_10_200_200_grid_data.mat')['state_input_to_net'], dtype = torch.float32)
    base_path = os.path.dirname(os.path.realpath(__file__))
    ensemble_forward = ensemble_forward_initialize(state_dim=STATE_DIM, control_dim=CONTROL_DIM, dyn_model_idx_list=dyn_model_idx_list, 
                                                    dyn_experiment_name='INV_PEND_DATA1H', base_path = base_path,
                                                    batch_size = 2000, if_gpu = True, model_checkpoint_epoch=model_checkpoint_epoch, mode = 'eval_without_ctrl')
    ensemble_dyn_forward_results = ensemble_forward(dyn_forward_samples=dyn_forward_samples)
    savemat('ensemble_dyn_forward_results/inv_pend_data1h_ensemble_forward_results.mat', ensemble_dyn_forward_results)
    print('Forward results saved.')