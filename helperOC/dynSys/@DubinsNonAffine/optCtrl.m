function uOpt = optCtrl(obj, ~, ~, deriv, uMode,~)
% uOpt = optCtrl(obj, t, y, deriv, uMode)

%% Input processing
if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

%% Optimal control
if strcmp(uMode, 'max')
  uOpt = (deriv{obj.dims==3}>=0)* (pi/2) + (deriv{obj.dims==3}<0)*(-pi/2);
elseif strcmp(uMode, 'min')
  uOpt = (deriv{obj.dims==3}>=0)* (-pi/2) + (deriv{obj.dims==3}<0)*(pi/2);
else
  error('Unknown uMode!')
end

end