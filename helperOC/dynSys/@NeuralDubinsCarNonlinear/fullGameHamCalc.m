function hamVal = fullGameHamCalc(obj, grid_size, uMode, dMode, deriv, f1_mean, f1_std, f2_mean, f2_std, dyn_training_stats)
    
    if ndims(f2_mean) == 2
        control_dim = 1;
    elseif ndims(f2_mean) == 3
        control_dim = size(f2_mean, 3);
    else
        error('Incorrect control dimension')
    end
    num_samples = size(f2_mean, 1);
    % account for cases with max/min deriv with only 1 element
    if size(deriv{1}) == 1
        deriv{1} = deriv{1} .* ones(grid_size(1), grid_size(2), grid_size(3));
        deriv{2} = deriv{2} .* ones(grid_size(1), grid_size(2), grid_size(3));
        deriv{3} = deriv{3} .* ones(grid_size(1), grid_size(2), grid_size(3));
    end
    
    deriv_x1_flattened = reshape(deriv{1}, [grid_size(1) * grid_size(2) * grid_size(3), 1]);
    deriv_x2_flattened = reshape(deriv{2}, [grid_size(1) * grid_size(2) * grid_size(3), 1]);
    deriv_x3_flattened = reshape(deriv{3}, [grid_size(1) * grid_size(2) * grid_size(3), 1]);
    deriv_hstacked = [deriv_x1_flattened, deriv_x2_flattened, deriv_x3_flattened];
  

    deriv_dot_mu = dot(deriv_hstacked, repmat(dyn_training_stats.labels_mean, size(deriv_hstacked,1), 1), 2); 
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

    ham_f1_term = dot(sigma_deriv, f1_mean, 2); % mu = train label mean
    ham_f1_d_term = dot(abs(sigma_deriv), f1_std, 2); % always positive

    hamVal = deriv_dot_mu + ham_f1_term;
    optCtrl = zeros(num_samples, control_dim);
    if strcmp(uMode, 'max') && strcmp(dMode, 'min')
        for i = 1:1:control_dim % compute opt ctrl for each control channel
            optCtrl(:,i) = (-abs_sigma_deriv_f2_std(:,i) < sigma_deriv_f2_mean(:,i) < abs_sigma_deriv_f2_std(:,i)) * 0 + ...
              (sigma_deriv_f2_mean(:,i) <= - abs_sigma_deriv_f2_std(:,i)) * obj.wRange(i,1) + ...
              (sigma_deriv_f2_mean(:,i) >= abs_sigma_deriv_f2_std(:,i)) * obj.wRange(i,2);  
        end
        ham_f2_d_term = dot(abs_sigma_deriv_f2_std, abs(optCtrl), 2);
        hamVal = hamVal - ham_f1_d_term - ham_f2_d_term; 
    elseif strcmp(uMode, 'min') && strcmp(dMode, 'max')
        for i = 1:1:control_dim
            optCtrl = (- abs_sigma_deriv_f2_std(:,i) < sigma_deriv_f2_mean(:,i) < abs_sigma_deriv_f2_std(:,i)) * 0 ...
                + (sigma_deriv_f2_mean(:,i) <= - abs_sigma_deriv_f2_std(:,i)) * obj.wRange(i,2) ...
                + (sigma_deriv_f2_mean(:,i) >= abs_sigma_deriv_f2_std(:,i)) * obj.wRange(i,1);
        end
        ham_f2_d_term = dot(abs_sigma_deriv_f2_std, abs(optCtrl), 2);
        hamVal = hamVal + ham_f1_d_term + ham_f2_d_term; 
    else
        error('unknown uMode')
    end
    ham_f2_u_term = dot(sigma_deriv_f2_mean, optCtrl, 2);
    hamVal = hamVal + ham_f2_u_term;

    hamVal = reshape(hamVal, [grid_size(1), grid_size(2), grid_size(3)]);

end