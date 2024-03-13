function hamValue = genericHam(t, data, deriv, schemeData)
% function hamValue = genericHam(t, data, deriv, schemeData)

%% Input unpacking
dynSys = schemeData.dynSys;

if ~isfield(schemeData, 'uMode')
  schemeData.uMode = 'min';
end

if ~isfield(schemeData, 'dMode')
  schemeData.dMode = 'max';
end

if ~isfield(schemeData, 'tMode')
  schemeData.tMode = 'backward';
end

% Custom derivative for MIE
if isfield(schemeData, 'deriv')
  deriv = schemeData.deriv;
end

%% Optimal control and disturbance

if isfield(schemeData, 'uIn')
  u = schemeData.uIn;
else
  u = dynSys.optCtrl(t, schemeData.grid.xs, deriv, schemeData);
end

if isfield(schemeData, 'dIn')
  d = schemeData.dIn;
else
  d = dynSys.optDstb(t, schemeData.grid.xs, deriv, u, schemeData);
end


hamValue = 0;
%% MIE
if isfield(schemeData, 'side')
  if strcmp(schemeData.side, 'lower')
%     if schemeData.dissComp
%       hamValue = hamValue - schemeData.dc;
%     end
%     
    TIderiv = -1;
  elseif strcmp(schemeData.side, 'upper')
%     if schemeData.dissComp
%       hamValue = hamValue + schemeData.dc;
%     end
    
%     hamValue = -hamValue;
%     deriv{1} = -deriv{1};
% %     if schemeData.trueMIEDeriv
% %       deriv{2} = -deriv{2};
% %     end
    TIderiv = 1;
  else
    error('Side of an MIE function must be upper or lower!')
  end
  
%   if ~schemeData.trueMIEDeriv
%     deriv(2) = computeGradients(schemeData.grid, data);
%   end
end

%% Plug optimal control into dynamics to compute Hamiltonian

% if schemeData.hamCompMode == 0
dx = dynSys.dynamics(t, schemeData.grid.xs, u, d, schemeData);
for i = 1:dynSys.nx
  hamValue = hamValue + deriv{i}.*dx{i};
end
% elseif schemeData.hamCompMode == 1 % simplified two player game
%     hamValue = dynSys.simplifiedGameHamCalc(size(schemeData.grid.xs{1}), schemeData.uMode, schemeData.dMode, deriv, schemeData.f1_mean, schemeData.f1_std, schemeData.f2_mean, schemeData.f2_std, schemeData.dyn_training_stats);
% elseif schemeData.hamCompMode == 2
%     hamValue = dynSys.fullGameHamCalc(size(schemeData.grid.xs{1}), schemeData.uMode, schemeData.dMode, deriv, schemeData.f1_mean, schemeData.f1_std, schemeData.f2_mean, schemeData.f2_std, schemeData.dyn_training_stats);
% else
%     error('Unrecognized ham computation mode');
% end

if isprop(dynSys, 'TIdim') && ~isempty(dynSys.TIdim)
  TIdx = dynSys.TIdyn(t, schemeData.grid.xs, u, d);
  hamValue = hamValue + TIderiv*TIdx{1};
end

%% Negate hamValue if backward reachable set
if strcmp(schemeData.tMode, 'backward')
  hamValue = -hamValue;
end

if isfield(schemeData, 'side')
  if strcmp(schemeData.side, 'upper')
    hamValue = -hamValue;
  end
end
end