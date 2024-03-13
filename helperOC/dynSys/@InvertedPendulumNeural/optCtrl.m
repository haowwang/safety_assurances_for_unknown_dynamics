function uOpt = optCtrl(obj, ~, x, deriv, schemeData)

    if ~iscell(deriv)
      deriv = num2cell(deriv);
    end
    
    % note: x is a cell array with nx number of cells. each cell is a dimension
    % of the mesh data
    
    f2_mean = schemeData.f2_mean;
    f2_std = schemeData.f2_std; 
    uMode = schemeData.uMode;
    dyn_training_stats = schemeData.dyn_training_stats; 
    control_dim = obj.nu;
    state_dim = obj.nx;
    grid_size = size(x{1});
    num_samples = 1;
    for i=1:1:length(grid_size)
        num_samples = num_samples * grid_size(i);
    end
    
    
    % account for cases with max/min deriv with only 1 element (in
    % genericPartial)
    if size(deriv{1},1) == 1
        for i=1:1:state_dim
            deriv{i} = deriv{i} .* ones(grid_size);
        end
    end

    deriv_hstacked = [];
    for i=1:1:state_dim
        deriv_hstacked = [deriv_hstacked, reshape(deriv{i}, num_samples, [])];
    end
    
    sigma_deriv = deriv_hstacked .* repmat(dyn_training_stats.labels_std, size(deriv_hstacked, 1), 1); % train label std (elt prod) dV/dx
    
    sigma_deriv_f2_mean = zeros(num_samples, control_dim);
    sigma_deriv_f2_std = zeros(num_samples, control_dim);
    for i=1:1:control_dim % iterate on each control dimension
        sigma_deriv_elt_wise_prod_f2_mean_column_sum = dot(sigma_deriv, f2_mean(:,:,i),2); % shape n x 1
        sigma_deriv_elt_wise_prod_f2_std_column_sum = dot(sigma_deriv, f2_std(:,:,i),2); % shape n x 1
        sigma_deriv_f2_mean(:,i) = sigma_deriv_elt_wise_prod_f2_mean_column_sum;
        sigma_deriv_f2_std(:,i) = sigma_deriv_elt_wise_prod_f2_std_column_sum;
    end
    
    abs_sigma_deriv_f2_std = abs(sigma_deriv_f2_std); 
    
    uOpt = cell(control_dim,1);
    if strcmp(uMode, 'max')
        for i = 1:1:control_dim % compute opt ctrl for each control channel
            uOpt{i} = (-abs_sigma_deriv_f2_std(:,i) < sigma_deriv_f2_mean(:,i) < abs_sigma_deriv_f2_std(:,i)) * 0 + ...
              (sigma_deriv_f2_mean(:,i) <= - abs_sigma_deriv_f2_std(:,i)) * obj.wRange(i,1) + ...
              (sigma_deriv_f2_mean(:,i) >= abs_sigma_deriv_f2_std(:,i)) * obj.wRange(i,2);
            uOpt{i} = reshape(uOpt{i}, grid_size); 
        end
    elseif strcmp(uMode, 'min')
        for i = 1:1:control_dim
            uOpt{i} = (- abs_sigma_deriv_f2_std(:,i) < sigma_deriv_f2_mean(:,i) < abs_sigma_deriv_f2_std(:,i)) * 0 ...
                + (sigma_deriv_f2_mean(:,i) <= - abs_sigma_deriv_f2_std(:,i)) * obj.wRange(i,2) ...
                + (sigma_deriv_f2_mean(:,i) >= abs_sigma_deriv_f2_std(:,i)) * obj.wRange(i,1);
            uOpt{i} = reshape(uOpt{i}, grid_size); 
        end
    else
        error('unknown uMode')
    end

end