function uOpt = optCtrl(obj, t, x, deriv, uMode, f2_output_mat,dyn_training_stats)
% uOpt = optCtrl(obj, t, y, deriv, uMode)

%% Input processing
if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

%% Optimal control
x1_size = size(x{1});
% account for cases with max/min deriv with only 1 element
if size(deriv{1}) == 1
    deriv{1} = deriv{1} .* ones(x1_size(1), x1_size(2), x1_size(3));
    deriv{2} = deriv{2} .* ones(x1_size(1), x1_size(2), x1_size(3));
    deriv{3} = deriv{3} .* ones(x1_size(1), x1_size(2), x1_size(3));
end

deriv_x1_flattened = reshape(deriv{1}, [x1_size(1) * x1_size(2) * x1_size(3), 1]);
x2_size = size(x{2});
deriv_x2_flattened = reshape(deriv{2}, [x2_size(1) * x2_size(2) * x2_size(3), 1]);
x3_size = size(x{3});
deriv_x3_flattened = reshape(deriv{3}, [x3_size(1) * x3_size(2) * x3_size(3), 1]);
deriv_hstacked = [deriv_x1_flattened, deriv_x2_flattened, deriv_x3_flattened];

f2_output_dot_deriv_stacked = sum(f2_output_mat .* deriv_hstacked, 2); % sum across each row
if strcmp(uMode, 'max')
%     uOpt = (deriv{obj.dims==3}>=0)* (pi/2) + (deriv{obj.dims==3}<0)*(-pi/2); % theta dot = sin(w)
%   uOpt = (deriv{obj.dims==3}>=0)*obj.wRange(2) +(deriv{obj.dims==3}<0)*(obj.wRange(1)); % theta dot = w
  uOpt = (f2_output_dot_deriv_stacked >= 0) *(obj.wRange(2)) + (f2_output_dot_deriv_stacked < 0)*obj.wRange(1);
  uOpt = reshape(uOpt, [x1_size(1), x1_size(2), x1_size(3)]);
elseif strcmp(uMode, 'min')
%     uOpt = (deriv{obj.dims==3}>=0)* (-pi/2) + (deriv{obj.dims==3}<0)*(pi/2); % theta dot = sin(w)
%   uOpt = (deriv{obj.dims==3}>=0)*(obj.wRange(1)) + (deriv{obj.dims==3}<0)*obj.wRange(2); % theta dot = w
  uOpt = (f2_output_dot_deriv_stacked >= 0) *(obj.wRange(1)) + (f2_output_dot_deriv_stacked < 0)*obj.wRange(2);
  uOpt = reshape(uOpt, [x1_size(1), x1_size(2), x1_size(3)]);
else
  error('Unknown uMode!')
end

end