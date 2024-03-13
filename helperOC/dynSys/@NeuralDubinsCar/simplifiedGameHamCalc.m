function hamVal = simplifiedGameHamCalc(obj, grid_size, uMode, dMode, deriv, f1_mean, f1_std, f2_mean, f2_std, dyn_training_stats)
    
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
  

    deriv_elt_wise_prod_mu = sum(deriv_hstacked .* repmat(dyn_training_stats.labels_mean, size(deriv_hstacked,1), 1), 2); 
    deriv_elt_wise_prod_sigma = deriv_hstacked .* repmat(dyn_training_stats.labels_std, size(deriv_hstacked, 1), 1);


    ham_f1_term = sum(deriv_elt_wise_prod_sigma .* f1_mean, 2); % mu = train label mean
    ham_f1_d_term = sum(abs(deriv_elt_wise_prod_sigma) .* f1_std, 2); % always positive

    if ndims(f2_mean) == 2 % 1 control case
        control_input = obj.wRange(2) * ones(grid_size(1) * grid_size(2) * grid_size(3), obj.nx); 
        ham_f2_u_term = sum(abs(deriv_elt_wise_prod_sigma) .* abs(f2_mean) .* control_input, 2); % always positive
        ham_f2_d_term = sum(abs(deriv_elt_wise_prod_sigma) .* abs(f2_std) .* control_input, 2); % always positive
    elseif ndims(f2_mean) > 2
        error('Multiple control case not implemented')
    end


    if strcmp(uMode, 'max') && strcmp(dMode, 'min')
        hamVal = deriv_elt_wise_prod_mu + ham_f1_term + ham_f2_u_term - ham_f1_d_term - ham_f2_d_term; 
    elseif strcmp(uMode, 'min') && strcmp(dMode, 'max')
        hamVal = deriv_elt_wise_prod_mu + ham_f1_term - ham_f2_u_term + ham_f1_d_term + ham_f2_d_term;
    else
      error('Unknown uMode!')
    end
    hamVal = reshape(hamVal, [grid_size(1), grid_size(2), grid_size(3)]);

end