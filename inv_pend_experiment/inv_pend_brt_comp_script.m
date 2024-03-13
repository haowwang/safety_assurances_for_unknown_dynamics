%% 
clear; close all; clc;

% set up the grid
load('dyn_datasets/inv_pend_data1h_dyn_dataset_summary.mat');
grid_min = [theta_bound(1), theta_dot_bound(1)];
grid_max = [theta_bound(2), theta_dot_bound(2)]; 
num_grid_pts_per_dim = 200;
N = [num_grid_pts_per_dim; num_grid_pts_per_dim]; % grid resolution
pdDims = 1; % periodic dimension (theta)
grid = createGrid(grid_min, grid_max, N, pdDims);
dyn_training_stats.labels_mean = labels_mean;
dyn_training_stats.labels_std = labels_std;

%% save grid point and ready to port to pytorch !!!only run this section if forward results have not been generated!!!
% x1_size = size(grid.xs{1});
% x1_flattened_vec = reshape(grid.xs{1}, [x1_size(1) * x1_size(2), 1]);
% x2_size = size(grid.xs{2});
% x2_flattened_vec = reshape(grid.xs{2}, [x2_size(1) * x2_size(2), 1]);
% state_input_to_net = [x1_flattened_vec, x2_flattened_vec];
% save("inv_pend_pi_10_200_200_grid_data.mat", "state_input_to_net");

%%
% load ensemble dyn forward results
load("ensemble_dyn_forward_results/inv_pend_data1h_ensemble_forward_results.mat");
schemeData.f1_mean = double(x_dot_f1_mean); % ensemble dynamics mean
schemeData.f1_std = 0. * double(x_dot_f1_std);
schemeData.f2_mean = double(x_dot_f2_mean); 
schemeData.f2_std = 0. * double(x_dot_f2_std);


%%
% target set information
target_set_center = [0.75*pi,0]; 
target_set_rectangle_length = [pi;5]; 
% data0 = shapeRectangleByCenter(grid, target_set_center, target_set_rectangle_length); 
target_set_theta_edge = 0.6 * pi;
normal = [-1; 0];
point = [target_set_theta_edge; 0];
hp0 = shapeHyperplane(grid, normal, point);
hp1 = shapeHyperplane(grid, -1 * normal, -1 * point);
data0 = shapeUnion(hp0, hp1);

% time 
t0 = 0; 
tMax = 0.6; 
dt = 0.05; 
tau = t0:dt:tMax; 

%%
initial_x = [0, 0]; 
inv_pend_params.u_min = control_bound(1);
inv_pend_params.u_max = control_bound(2);
inv_pend_params.l = l;
inv_pend_params.m = m; 
inv_pend_params.g = g;
inv_pend_params.b = b;
% inv_pend = InvertedPendulum(initial_x, inv_pend_params); % analytical inv pend
inv_pend = NeuralNetDyn(initial_x, control_bound, 1:2, 1); % neural inv pend

% problem parameters
schemeData.dynSys = inv_pend; 
schemeData.uMode = 'max';
schemeData.dMode = 'min';
schemeData.tMode = 'backward'; 
schemeData.accuracy = 'high';
schemeData.grid = grid;
schemeData.dyn_training_stats = dyn_training_stats; % labels mean and std, needed to unnormalize dynamics output
compMethod = 'minVOverTime'; % objective of optimal control is min over time (BRT instead of BRS)

HJIextraArgs.visualize.valueSet = 1; % visualize the BRT
HJIextraArgs.visualize.initialValueSet = 1; % visualize the initial BRT
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; 


tic
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, compMethod, HJIextraArgs);
toc