function dx = dynamics(obj, ~, x, u, d,f1_output, f2_output_mat, dyn_training_stats)
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

if iscell(x)
  dx = cell(length(obj.dims), 1);
  
  for i = 1:length(obj.dims)
    dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
  end
else
  dx = zeros(obj.nx, 1);
  dx(1) = 5 * sin(x(1)) .* cos(x(3)) + d(1);
  dx(2) = 5 * cos(x(2)) .* sin(x(3)) + d(2);
  dx(3) = sin(x(1)) .* cos(x(2)) .* u + d(3);
end
end

function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)

switch dim
  case 1
    dx = 5 .* sin(x{dims==1}) .* cos(x{dims==3}) + d{1};
  case 2
    dx = 5 .* cos(x{dims==2}) .* sin(x{dims==3}) + d{2};
  case 3
    dx = sin(x{dims==1}) .* cos(x{dims==2}) .* u + d{3};
  otherwise
    error('Only dimension 1-3 are defined for dynamics of DubinsCar!')
end
end