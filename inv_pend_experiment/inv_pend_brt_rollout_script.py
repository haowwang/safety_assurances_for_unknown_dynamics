import numpy as np
import torch
import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append( os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))
sys.path.append( os.path.dirname(os.path.dirname(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )))
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from utils import ensemble_forward_initialize

def get_matlab_variables(mat_file_path):
   variables = loadmat(mat_file_path)
   val_func_data = variables['data']
   deriv_theta = variables['deriv'][0][0]
   deriv_theta_dot = variables['deriv'][1][0]
   coordinates = variables['grid'][0][0][6]
   theta_coord = coordinates[0][0].squeeze()
   theta_dot_coord = coordinates[1][0].squeeze()
   time = variables['tau'].squeeze()
   time_horizon = time[-1]
   theta_edge = variables['target_set_theta_edge'].squeeze()
   matlab_var_dict = dict(val_func_data=val_func_data,
                        deriv_theta = deriv_theta,
                        deriv_theta_dot = deriv_theta_dot,
                        theta_coord=theta_coord,
                        theta_dot_coord=theta_dot_coord,
                        time=time,
                        time_horizon = time_horizon,
                        theta_edge = theta_edge
                        )
   return matlab_var_dict


def get_val_func_eval_function(matlab_var_dict):
   val_func_eval = RegularGridInterpolator((matlab_var_dict['theta_coord'], matlab_var_dict['theta_dot_coord'], matlab_var_dict['time']), matlab_var_dict['val_func_data'],  bounds_error=False, fill_value=None)
   return val_func_eval

def get_val_func_theta_deriv_func_eval_function(matlab_var_dict):
   val_func_theta_deriv_eval = RegularGridInterpolator((matlab_var_dict['theta_coord'], matlab_var_dict['theta_dot_coord'], matlab_var_dict['time']), matlab_var_dict['deriv_theta'] ,  bounds_error=False, fill_value=None)
   return val_func_theta_deriv_eval

def get_val_func_theta_dot_deriv_func_eval_function(matlab_var_dict):
   val_func_theta_dot_deriv_eval = RegularGridInterpolator((matlab_var_dict['theta_coord'], matlab_var_dict['theta_dot_coord'], matlab_var_dict['time']), matlab_var_dict['deriv_theta_dot'] ,  bounds_error=False, fill_value=None)
   return val_func_theta_dot_deriv_eval


def get_val_func_grad_initialize(val_func_theta_deriv_eval, val_func_theta_dot_deriv_eval):
   def get_val_func_grad(states, time):
      """
      batched gradients
      """
      batch_size = states.shape[0]
      state_dim = states.shape[1]
      dVx = np.zeros((batch_size, state_dim))
      for i in range(batch_size):
         state = states[[i], :] # shape 1 x 2
         state_time = np.hstack((state, time))
         d_theta = val_func_theta_deriv_eval(state_time) # shape (1,)
         d_theta_dot = val_func_theta_dot_deriv_eval(state_time) # shape (1,)
         dVx[i,0] = d_theta.item()
         dVx[i,1] = d_theta_dot.item()
      return dVx
   return get_val_func_grad


def inv_pend_dyn_initialize(m,l,b,g):
    """
    x0 = theta, x1 = theta dot
    u = torque 
    batched ground truth dynamics
    """
    def inv_pend_dyn(x,u):
        x_dot_0 = x[:, [1]] # x0 dot = theta dot = x1
        x_dot_1 = (-b * x[:, [1]] + (m*g*l*np.sin(x[:, [0]]))/2 - u) / ((m*l**2)/3)
        x_dot = np.hstack((x_dot_0, x_dot_1))
        return x_dot
    return inv_pend_dyn


def inv_pend_ensemble_dyn_opt_control_initialize(train_labels_std, control_mode, control_range):
   def inv_pend_ensemble_dyn_opt_control(deriv, f2_mean, f2_std):
      """
      batched optimal control for learned dynamics
      deriv shape batch_size x Dx (n x 2)
      f2_mean shape batch_size x Dx x Du (n x 2 x 1)
      f2_std shape batch_size x Dx x Du (n x 2 x 1)
      """
      control_dim = f2_std.shape[2]
      batch_size = f2_std.shape[0]
      sigma_deriv = train_labels_std * deriv
      sigma_deriv = sigma_deriv[:,np.newaxis, :] # shape batch_size x 1 x Dx

      sigma_deriv_f2_mean = np.matmul(sigma_deriv, f2_mean) # shape batch_size x 1 x Du
      sigma_deriv_f2_mean = sigma_deriv_f2_mean.squeeze(1) # shape batch_size x Du
      sigma_deriv_f2_std = np.matmul(sigma_deriv, f2_std) # shape batch_size x 1 x Du
      sigma_deriv_f2_std = sigma_deriv_f2_std.squeeze(1) # shape batch_size x Du
      abs_sigma_deriv_f2_std = np.abs(sigma_deriv_f2_std)
      opt_ctrl = np.zeros((batch_size,control_dim))
      if control_mode == 'min':
         for i in range(control_dim):
            opt_ctrl[:,[i]] = (np.logical_and(-abs_sigma_deriv_f2_std[:,[i]] < sigma_deriv_f2_mean[:,[i]], sigma_deriv_f2_mean[:,[i]] < abs_sigma_deriv_f2_std[:,[i]])) * 0 + \
            (sigma_deriv_f2_mean[:,i] <= - abs_sigma_deriv_f2_std[:,i]) * control_range[i,1] + \
            (sigma_deriv_f2_mean[:,i] >= abs_sigma_deriv_f2_std[:,i]) * control_range[i,0] 
      elif control_mode == 'max':
         for i in range(control_dim):
            opt_ctrl[:,[i]] = (np.logical_and(-abs_sigma_deriv_f2_std[:,[i]] < sigma_deriv_f2_mean[:,[i]], sigma_deriv_f2_mean[:,[i]] < abs_sigma_deriv_f2_std[:,[i]])) * 0 + \
            (sigma_deriv_f2_mean[:,[i]] <= - abs_sigma_deriv_f2_std[:,[i]]) * control_range[i,0] + \
            (sigma_deriv_f2_mean[:,[i]] >= abs_sigma_deriv_f2_std[:,[i]]) * control_range[i,1] 
      else:
         sys.exit('Unknown control mode. Exit')
      return opt_ctrl 
   return inv_pend_ensemble_dyn_opt_control
   

# def dubins3d_default_opt_control_initialize(control_mode, control_range):
#    def dubins3d_default_opt_control(deriv,f2_mean, f2_std):
#       theta_deriv = deriv[0,2]
#       if control_mode == 'min':
#          opt_ctrl = (theta_deriv <= 0) * control_range[0,1] + (theta_deriv > 0) * control_range[0,0]
#       elif control_mode == 'max':
#          opt_ctrl = (theta_deriv <= 0) * control_range[0,0] + (theta_deriv > 0) * control_range[0,1]
#       else:
#          sys.exit('Unknown control mode. Exit')
#       print(opt_ctrl)
#       return opt_ctrl
#    return dubins3d_default_opt_control         
   
   
def inv_pend_rollout_initialize(time_horizon_init, delta_t, ground_truth_dyn, ensemble_dyn_forward, opt_ctrl_func, deriv_func):
   def inv_pend_rollout(states_init):
      batch_size = states_init.shape[0]
      time_horizon = time_horizon_init
      states_arr = states_init[:, np.newaxis, :] # shape batch_size x 1 x 2
      controls_arr = np.zeros((batch_size,1))
      while time_horizon > 0:
         current_states = states_arr[:, -1, :] # shape batch_size x 1 x 2
         val_func_grads = deriv_func(states = current_states, time = np.array([[time_horizon]])) # shape batch_size x 2 
         ensemble_forward_results = ensemble_dyn_forward(dyn_forward_samples = torch.tensor(current_states, dtype = torch.float32))
         ensemble_f2_mean = ensemble_forward_results['x_dot_f2_mean'] # shape batch_size x Dx x Du
         ensemble_f2_std = ensemble_forward_results['x_dot_f2_std'] # shape batch_size x Dx x Du
         opt_ctrls = opt_ctrl_func(deriv = val_func_grads, f2_mean = ensemble_f2_mean, f2_std = ensemble_f2_std)
         states_dot = ground_truth_dyn(x = current_states, u = opt_ctrls)
         next_states = current_states + delta_t * states_dot # shape batch_size x Dx
         next_states = next_states[:, np.newaxis, :] # shape batch_size x 1 x Dx
         states_arr = np.concatenate((states_arr, next_states), axis = 1)
         controls_arr = np.hstack((controls_arr, opt_ctrls))
         time_horizon -= delta_t
      return states_arr
   return inv_pend_rollout


def if_enter_target_set_initialize(theta_edge):
   def if_enter_target_set(states_arr):
      theta_arr = states_arr[:, :, 0]
      bool_arr = np.abs(theta_arr) > theta_edge
      return np.sum(bool_arr, axis = 1, keepdims=True) > 0
   return if_enter_target_set


def inv_pend_verification(N, M, theta_bound, theta_dot_bound, time_horizon, inv_pend_rollout, inv_pend_val_func, if_enter_target_set_func, batch_size):
   delta = 0
   delta_intermediate = 0
   for i in range(M):
      violation_flag = 0
      j = 0
      batched_rollout_states_init = None
      batched_rollout_states_value = None
      while j < N:
         sampled_state = np.array([[np.random.uniform(low=theta_bound[0], high=theta_bound[1]), np.random.uniform(low=theta_dot_bound[0], high=theta_dot_bound[1])]])
         sampled_state_value = inv_pend_val_func(np.hstack((sampled_state, np.array([[time_horizon]]))))
         if batched_rollout_states_init is None and sampled_state_value > delta:
               batched_rollout_states_init = sampled_state
               batched_rollout_states_value = np.array([[sampled_state_value]])
               j += 1
         elif batched_rollout_states_init is not None and batched_rollout_states_init.shape[0] < batch_size and sampled_state_value > delta:
               batched_rollout_states_init = np.vstack((batched_rollout_states_init, sampled_state))
               batched_rollout_states_value = np.vstack((batched_rollout_states_value, np.array([[sampled_state_value]])))
               j += 1
         print(i, j, delta, delta_intermediate)
         if batched_rollout_states_init is not None:  
            if batched_rollout_states_init.shape[0] == batch_size or j == N - 1:
               rollout_states = inv_pend_rollout(states_init = batched_rollout_states_init)
               if_enter_target_set_bool = if_enter_target_set_func(states_arr = rollout_states)
               if np.any(if_enter_target_set_bool):
                  violation_flag = 1
                  violation_states_value = batched_rollout_states_value[if_enter_target_set_bool]
                  max_violation_state_value = np.max(violation_states_value)
                  if max_violation_state_value > delta_intermediate:
                     delta_intermediate = max_violation_state_value
               batched_rollout_states_init = None # reset for next batch
               batched_rollout_states_value = None # reset for next batch
      if not violation_flag: # not sample state rollout enter target set. exit 
         return delta
      delta = delta_intermediate
   return delta



def main():
   STATE_DIM = 2
   CONTROL_DIM = 1
   base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
   dyn_model_idx_list = range(5)
   dataset_summary = loadmat('dyn_datasets/inv_pend_full_ss_deficit_5h_dyn_dataset_summary')
   l = dataset_summary['l']
   m = dataset_summary['m']
   g = dataset_summary['g']
   b = dataset_summary['b']
   theta_bound = dataset_summary['theta_bound'].squeeze()
   theta_dot_bound = dataset_summary['theta_dot_bound'].squeeze()
   train_labels_std = dataset_summary['labels_std']
   control_bound = dataset_summary['control_bound']

   matlab_var_dict = get_matlab_variables('inv_pend_5h_0std_avoid_brt_data.mat')
   time_horizon = matlab_var_dict['time_horizon']
   batch_size = 1024
   ensemble_forward = ensemble_forward_initialize(state_dim=STATE_DIM, control_dim=CONTROL_DIM, dyn_model_idx_list=dyn_model_idx_list, 
                                                      dyn_experiment_name='INV_PEND_FD_DATA5H_E5', base_path = base_path, if_gpu=True,
                                                      batch_size = batch_size)

   val_func_theta_deriv_eval = get_val_func_theta_deriv_func_eval_function(matlab_var_dict)
   val_func_theta_dot_deriv_eval = get_val_func_theta_dot_deriv_func_eval_function(matlab_var_dict)
   val_func_eval = get_val_func_eval_function(matlab_var_dict)

   dyn = inv_pend_dyn_initialize(m=m,l=l,b=b,g=g)
   opt_ctrl_func = inv_pend_ensemble_dyn_opt_control_initialize(train_labels_std=train_labels_std, control_mode = 'max', control_range = control_bound)
   # dubins3d_default_opt_ctrl_func = dubins3d_default_opt_control_initialize(control_mode='min', control_range=dubins3d_control_bound)
   val_func_deriv_func = get_val_func_grad_initialize(val_func_theta_deriv_eval, val_func_theta_dot_deriv_eval)
   inv_pend_rollout_func = inv_pend_rollout_initialize(time_horizon_init = time_horizon, delta_t = 0.01, 
                                          ground_truth_dyn = dyn, ensemble_dyn_forward = ensemble_forward, 
                                          opt_ctrl_func = opt_ctrl_func, deriv_func = val_func_deriv_func)
   val_func_eval = get_val_func_eval_function(matlab_var_dict=matlab_var_dict)
   if_enter_target_set_func = if_enter_target_set_initialize(theta_edge=matlab_var_dict['theta_edge'])
   inv_pend_verification(N = 2763, M = 10, theta_bound = theta_bound, theta_dot_bound = theta_dot_bound, 
                         time_horizon = time_horizon, inv_pend_rollout = inv_pend_rollout_func, inv_pend_val_func = val_func_eval, 
                         if_enter_target_set_func = if_enter_target_set_func, batch_size = batch_size)

   # plt.figure()
   # ax = plt.gca()
   # ax.plot(state_arr[:,0], state_arr[:,1])
   # plt.axvline(x = 0.6*np.pi, color = 'g')
   # plt.axvline(x = -0.6*np.pi, color = 'g')
   # # target_set = plt.Circle((0,0), 0.25)
   # # ax.add_patch(target_set)
   # # ax.set_aspect('equal', 'box')
   # plt.show()

if __name__ == "__main__":
   main()