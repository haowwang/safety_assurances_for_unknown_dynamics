function dx = dynamics(obj, ~, x, u, d, schemeData)
% dynamics for learned neural dynamics

    if ~iscell(d) && isempty(d) % don't use dstb
        f1_full = schemeData.f1_mean;
        f2_full = schemeData.f2_mean;
    elseif iscell(d)
        d1 = d{1};
        d2 = d{2};
        f1_full = schemeData.f1_mean + d1;
        f2_full = schemeData.f2_mean + d2;
    end

    dyn_training_stats = schemeData.dyn_training_stats; 
    num_samples = size(schemeData.f1_mean, 1);
    state_dim = obj.nx;
    control_dim = obj.nu; % u is a cell array
    
    dx = cell(state_dim, 1);
    
    grid_size = size(x{1}); % placeholder for grid size
    f2_full_dot_u = zeros(num_samples, state_dim); % (f2(x)+d2(x))u

    for i=1:1:control_dim
        i_th_dim_control = reshape(u{i}, num_samples,[]); % reshape to a vector of size n x 1 
         f2_full_dot_u = f2_full_dot_u + f2_full(:,:,i) .* repmat(i_th_dim_control, 1, state_dim); % vector of shape n x Dx
    end

    x_dot = f1_full + f2_full_dot_u; 
    % unnormalize net output
    x_dot = x_dot .* repmat(dyn_training_stats.labels_std, num_samples, 1) + repmat(dyn_training_stats.labels_mean, num_samples, 1);
    for i = 1:length(obj.dims)
        dx{i} = reshape(x_dot(:,i),grid_size); % assume grid resolution is same across dims
    end
end