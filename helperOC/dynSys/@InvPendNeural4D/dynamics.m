function dx = dynamics(obj, ~, x, u, d, f1_output, f2_output_mat, dyn_training_stats)
% Dynamics of the Dubins Car
%    \dot{x}_1 = v * cos(x_3) + d1
%    \dot{x}_2 = v * sin(x_3) + d2
%    \dot{x}_3 = w
%   Control: u = w;
%
% Mo Chen, 2016-06-08

    if nargin < 5
      d = [0; 0; 0];
    end

    dx = cell(length(obj.dims), 1);
    x1_size = size(x{1});
    control_size = size(u);
    if size(u, 1) == 1 % account for cases where the function is involed with a scalar control
        control_input_to_net = u * ones(x1_size(1) * x1_size(2) * x1_size(3) * x1_size(4), 1);
    else
        control_input_to_net = reshape(u, [control_size(1) * control_size(2) * control_size(3) * control_size(4), 1]);
    end
    dx_from_net = InvPendNet(f1_output, f2_output_mat, control_input_to_net, dyn_training_stats);
    
    % dynamics clipping
    dyn_max = repmat(dyn_training_stats.dyn_max,size(dx_from_net,1),1);
    dyn_min = repmat(dyn_training_stats.dyn_min, size(dx_from_net,1),1);
    dx_from_net = min(dyn_max, max(dyn_min, dx_from_net));

    for i = 1:length(obj.dims)
        dx{i} = reshape(dx_from_net(:,i),[x1_size(1), x1_size(2), x1_size(3), x1_size(4)]); % assume grid resolution is same across dims
    end
end




