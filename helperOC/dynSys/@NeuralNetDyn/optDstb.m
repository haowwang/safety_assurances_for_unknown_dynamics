function dOpt = optDstb(obj, ~, x, deriv, uOpt, schemeData)

    state_dim = obj.nx;
    control_dim = obj.nu;
    f1_std = schemeData.f1_std;
    f2_std = schemeData.f2_std;
    dyn_training_stats = schemeData.dyn_training_stats;
    dMode = schemeData.dMode;
    grid_size = size(x{1});
    num_samples = 1;
    for i=1:1:length(grid_size)
        num_samples = num_samples * grid_size(i);
    end
    
    % case of max/min derivative (only 1 deriv vector) for genericPartial
    % computation
    if size(deriv{1},1) == 1
        for i=1:1:state_dim
            deriv{i} = deriv{i} .* ones(grid_size);
        end
    end
    
    deriv_hstacked = [];
    for i=1:1:state_dim
        deriv_hstacked = [deriv_hstacked, reshape(deriv{i}, num_samples, [])];
    end

    sigma_deriv = deriv_hstacked .* repmat(dyn_training_stats.labels_std, num_samples, 1); % train label std (elt prod) dV/dx
    
    % compute d2 each control dimension at a time
    d2 = zeros(num_samples, state_dim, control_dim);
    if strcmp(dMode, 'min')
        for i=1:1:control_dim
            i_th_dim_control = reshape(uOpt{i}, num_samples, []); % shape num_samples by 1
            sigma_deriv_elt_prod_i_th_dim_control = sigma_deriv .* repmat(i_th_dim_control, 1, state_dim); % shape num_samples by state_dim
            d2(:,:,i) = (sigma_deriv_elt_prod_i_th_dim_control <= 0) .* f2_std(:,:,i) + (sigma_deriv_elt_prod_i_th_dim_control > 0) .* (-f2_std(:,:,i));
        end
        d1 = (sigma_deriv <= 0) .* f1_std + (sigma_deriv > 0) .* (-f1_std);
    elseif strcmp(dMode, 'max')
        for i=1:1:control_dim
            i_th_dim_control = reshape(uOpt{i}, num_samples, []); % shape num_samples by 1
            sigma_deriv_elt_prod_i_th_dim_control = sigma_deriv .* repmat(i_th_dim_control, 1, state_dim); % shape num_samples by state_dim
            d2(:,:,i) = (sigma_deriv_elt_prod_i_th_dim_control <= 0) .* (-f2_std(:,:,i)) + (sigma_deriv_elt_prod_i_th_dim_control > 0) .* f2_std(:,:,i);
        end
        d1 = (sigma_deriv <= 0) .* (-f1_std) + (sigma_deriv > 0) .* f1_std;
    end
    
    dOpt{1} = d1;
    dOpt{2} = d2; 
end