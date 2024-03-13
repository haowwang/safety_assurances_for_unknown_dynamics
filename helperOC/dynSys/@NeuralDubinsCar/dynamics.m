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
    control_size = size(u);
    x1_size = size(x{1});
    if size(u, 1) == 1 % account for cases where the function is involed with a scalar control
        control_input_to_net = u * ones(x1_size(1) * x1_size(2) * x1_size(3), 1);
    else
      control_input_to_net = reshape(u, [control_size(1) * control_size(2) * control_size(3), 1]);
    end
    if ndims(f2_output_mat) == 2 % one control case
        control_input_to_net = repmat(control_input_to_net,1,3);
        f2_output = f2_output_mat .* control_input_to_net;
    elseif ndims(f2_output_mat) == 3 % multiple controls case
        disp('multiple control not implemented')
    end
    x_dot = f1_output + f2_output;
    % unnormalize net output
    x_dot = x_dot .* repmat(dyn_training_stats.labels_std, size(x_dot, 1), 1) + repmat(dyn_training_stats.labels_mean, size(x_dot,1), 1);
    for i = 1:length(obj.dims)
        dx{i} = reshape(x_dot(:,i),[x1_size(1), x1_size(2), x1_size(3)]); % assume grid resolution is same across dims
    end
end



% function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)
% 
%     switch dim
%       case 1
%         dx = obj.speed * cos(x{dims==3}) + d{1};
%       case 2
%         dx = obj.speed * sin(x{dims==3}) + d{2};
%       case 3
%         dx = u + d{3};
%       otherwise
%         error('Only dimension 1-3 are defined for dynamics of DubinsCar!')
%     end
% end




