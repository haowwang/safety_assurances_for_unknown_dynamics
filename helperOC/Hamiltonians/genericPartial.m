function alpha = genericPartial(t, data, derivMin, derivMax, schemeData, dim)
% alpha = genericPartial(t, data, derivMin, derivMax, schemeData, dim)

g = schemeData.grid;
dynSys = schemeData.dynSys;

if ismethod(dynSys, 'partialFunc')
%   disp('Using partial function from dynamical system')
  alpha = dynSys.partialFunc(t, data, derivMin, derivMax, schemeData, dim);
  return
end

if ~isfield(schemeData, 'uMode')
  schemeData.uMode = 'min';
end

if ~isfield(schemeData, 'dMode')
  schemeData.dMode = 'min';
end

% TIdim = [];
% dims = 1:dynSys.nx;
% if isfield(schemeData, 'MIEdims')
%   TIdim = schemeData.TIdim;
%   dims = schemeData.MIEdims;
% end

% x = cell(dynSys.nx, 1);
% x(dims) = g.xs;

%% Compute control
if isfield(schemeData, 'uIn')
  % Control
  uU = schemeData.uIn;
  uL = schemeData.uIn;
 
else
  % Optimal control assuming maximum deriv
  uU = dynSys.optCtrl(t, g.xs, derivMax, schemeData); 
  
  % Optimal control assuming minimum deriv
  uL = dynSys.optCtrl(t, g.xs, derivMin, schemeData); 
end

%% Compute disturbance
if isfield(schemeData, 'dIn')
  dU = schemeData.dIn;
  dL = schemeData.dIn;
  
else
  dU = dynSys.optDstb(t, g.xs, derivMax, uU, schemeData);
  dL = dynSys.optDstb(t, g.xs, derivMin, uL, schemeData);
end
  
%% Compute alpha
% dxUU = dynSys.dynamics(t, schemeData.grid.xs, uU, dU);
% dxUL = dynSys.dynamics(t, schemeData.grid.xs, uU, dL);
% dxLL = dynSys.dynamics(t, schemeData.grid.xs, uL, dL);
% dxLU = dynSys.dynamics(t, schemeData.grid.xs, uL, dU);
dxUU = dynSys.dynamics(t, schemeData.grid.xs, uU, dU, schemeData);
dxUL = dynSys.dynamics(t, schemeData.grid.xs, uU, dL, schemeData);
dxLL = dynSys.dynamics(t, schemeData.grid.xs, uL, dL, schemeData);
dxLU = dynSys.dynamics(t, schemeData.grid.xs, uL, dU, schemeData);
alpha = max(abs(dxUU{dim}), abs(dxUL{dim}));
alpha = max(alpha, abs(dxLL{dim}));
alpha = max(alpha, abs(dxLU{dim}));
end
