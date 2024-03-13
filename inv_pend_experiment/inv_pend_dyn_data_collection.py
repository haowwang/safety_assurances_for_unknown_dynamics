import numpy as np 
from scipy.io import savemat

def inv_pend_dyn_initialize(m,l,b,g):
    """
    x0 = theta, x1 = theta dot
    u = torque 
    """
    def inv_pend_dyn(x,u):
        x_dot_0 = x[:, [1]] # x0 dot = theta dot = x1
        x_dot_1 = (-b * x[:, 1] + (m*g*l*np.sin(x[:, 0]))/2 - u) / ((m*l**2)/3)
        x_dot = np.hstack((x_dot_0, x_dot_1))
        return x_dot
    return inv_pend_dyn


def inv_pend_rollout_stopping_condition(x_arr):
    x_t = x_arr[[-1], :]
    theta_t = x_t[0,0]
    if np.abs(theta_t) > np.pi:
        return True
    else:
        return False


def inv_pend_u_gen_initialize(u_bound):
    def inv_pend_u_gen(size):
        u = np.random.uniform(low = u_bound[0], high = u_bound[1], size = size)
        return u
    return inv_pend_u_gen


def inv_pend_rollout(x0, dyn, u_gen, stopping_condition):
    """
    stopping condition is a function, u_gen is also a function
    """
    delta_t = 0.01
    x_arr = x0.reshape(1,2)
    u_arr = None
    x_dot_arr = None
    assert x0.shape[0] == 1 and x0.shape[1] == 2 # enforce shape of starting state
    while True:
        if stopping_condition(x_arr): # stopping condition based on state or number of steps in the rollout
            x_u_arr = np.hstack((x_arr[:-1], u_arr))
            return x_u_arr[:,], x_dot_arr[:,]
        x_t = x_arr[[-1],:]
        u_t = u_gen(size=(1,1))
        x_dot = dyn(x_t, u_t)
        x_next = x_t + x_dot * delta_t
        x_arr = np.vstack((x_arr, x_next))
        if u_arr is None:
            u_arr = u_t
            x_dot_arr = x_dot
        else:
            u_arr = np.vstack((u_arr, u_t))
            x_dot_arr = np.vstack((x_dot_arr, x_dot))


def inv_pend_dyn_gen_rollout(theta_bound, theta_dot_bound, inv_pend_dyn, inv_pend_u_gen, num_samples, stopping_condition):
    
    x_u_arr = np.zeros((1,3))
    x_dot_arr = np.zeros((1,2))
    while True:
        if x_u_arr.shape[0] > num_samples+1:
            return x_u_arr[1:num_samples+1, :], x_dot_arr[1:num_samples+1, :]
        random_x_init = np.array([[np.random.uniform(low=theta_bound[0], high = theta_bound[1]), 
                                   np.random.uniform(low=theta_dot_bound[0], high = theta_dot_bound[1])]])
        x_u_from_rollout, x_dot_from_rollout = inv_pend_rollout(x0 = random_x_init, dyn=inv_pend_dyn, u_gen = inv_pend_u_gen, stopping_condition=stopping_condition)
        x_u_arr = np.vstack((x_u_arr, x_u_from_rollout))
        x_dot_arr = np.vstack((x_dot_arr, x_dot_from_rollout))


def dyn_dataset_stats_summary(dataset):
    train_inputs = dataset['train_inputs']
    train_labels = dataset['train_labels']
    num_states = train_labels.shape[1]
    train_inputs_max = np.max(train_inputs[:,:num_states], axis = 0, keepdims = True)
    train_inputs_min = np.min(train_inputs[:,:num_states], axis = 0, keepdims = True)
    train_labels_max = np.max(train_labels, axis = 0, keepdims = True)
    train_labels_min = np.min(train_labels, axis = 0, keepdims = True)
    inputs_mean = np.mean(train_inputs[:,:num_states], axis = 0, keepdims=True)
    inputs_std = np.std(train_inputs[:,:num_states], axis = 0, keepdims=True)
    labels_means = np.mean(train_labels, axis = 0, keepdims=True)
    labels_std = np.std(train_labels, axis = 0, keepdims=True)
    training_data_stats = {'inputs_mean':inputs_mean, 'inputs_std':inputs_std, 'labels_mean':labels_means, 'labels_std':labels_std, 'train_inputs_min':train_inputs_min, 
                           'train_inputs_max':train_inputs_max, 'train_labels_min':train_labels_min, 'train_labels_max':train_labels_max, 'num_train_samples':train_inputs.shape[0]}
    return training_data_stats


def inv_pend_experiment_data_generation():
    m = 0.5
    l = 1.
    b = 0.1
    g = 9.8
    u_bound = [-0.25, 0.25]
    theta_bound = [-np.pi, np.pi] 
    theta_dot_bound = [-10., 10.]
    inv_pend_dyn = inv_pend_dyn_initialize(m=m, l=l, b=b, g=g)
    inv_pend_u_gen = inv_pend_u_gen_initialize(u_bound = u_bound)
    x_u_arr_train, x_dot_arr_train = inv_pend_dyn_gen_rollout(theta_bound = theta_bound, theta_dot_bound = theta_dot_bound, inv_pend_dyn=inv_pend_dyn,
                                                              inv_pend_u_gen = inv_pend_u_gen, num_samples=100, stopping_condition=inv_pend_rollout_stopping_condition)
    x_u_arr_eval, x_dot_arr_eval = inv_pend_dyn_gen_rollout(theta_bound = theta_bound, theta_dot_bound = theta_dot_bound, inv_pend_dyn=inv_pend_dyn,
                                                              inv_pend_u_gen = inv_pend_u_gen, num_samples=5000, stopping_condition=inv_pend_rollout_stopping_condition)
    dataset = {'train_inputs':x_u_arr_train, 'train_labels':x_dot_arr_train, 'eval_inputs':x_u_arr_eval, 'eval_labels':x_dot_arr_eval, 'theta_bound':theta_bound,
               'theta_dot_bound':theta_dot_bound, 'control_bound':u_bound, 'm':m, 'l':l, 'b':b, 'g':g}
    dyn_dataset_stats_summary_dict = dyn_dataset_stats_summary(dataset=dataset)
    dyn_dataset_stats_summary_dict['theta_dot_bound'] = theta_dot_bound
    dyn_dataset_stats_summary_dict['theta_bound'] = theta_bound
    dyn_dataset_stats_summary_dict['control_bound'] = u_bound
    dyn_dataset_stats_summary_dict['m'] = m
    dyn_dataset_stats_summary_dict['l'] = l
    dyn_dataset_stats_summary_dict['g'] = g
    dyn_dataset_stats_summary_dict['b'] = b
    dataset_name = 'inv_pend_data1h_dyn_dataset'
    savemat('dyn_datasets/' + dataset_name + '.mat', dataset)
    savemat('dyn_datasets/' + dataset_name + '_summary.mat', dyn_dataset_stats_summary_dict)
    print('Dataset and dataset summary saved.')


if __name__ == "__main__":
    inv_pend_experiment_data_generation()


    

