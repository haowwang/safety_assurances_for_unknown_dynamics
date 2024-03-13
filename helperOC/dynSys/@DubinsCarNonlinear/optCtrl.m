function uOpt = optCtrl(obj, ~, grid_x, deriv, uMode,f2_output_mat, dyn_training_stats)
% uOpt = optCtrl(obj, t, y, deriv, uMode)

%% Input processing
% if nargin < 5
%   uMode = 'min';
% end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

%% Optimal control
if strcmp(uMode, 'max')
  uOpt = ((sin(grid_x{1}) .* cos(grid_x{2}) .* deriv{obj.dims==3})>=0)*obj.wRange(2) + ((sin(grid_x{1}) .* cos(grid_x{2}) .* deriv{obj.dims==3}) < 0)*obj.wRange(1);
elseif strcmp(uMode, 'min')
  uOpt = ((sin(grid_x{1}) .* cos(grid_x{2}) .* deriv{obj.dims==3})>=0)*obj.wRange(1) + ((sin(grid_x{1}) .* cos(grid_x{2}) .* deriv{obj.dims==3}) < 0)*obj.wRange(2);
else
  error('Unknown uMode!')
end

end