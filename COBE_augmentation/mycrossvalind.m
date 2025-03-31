function group_labels = mycrossvalind(n, k)
    % n: num_samples
    % k: num_folds
    
    sample_indices = 1:n;
    
    samples_per_group = floor(n / k);
    remainder = mod(n, k);
    
    group_sizes = repmat(samples_per_group, 1, k);
    group_sizes(1:remainder) = group_sizes(1:remainder) + 1;
    
    shuffled_indices = sample_indices(randperm(n));
    group_labels = zeros(1, n);
    
    start_index = 1;
    for i = 1:k
        end_index = start_index + group_sizes(i) - 1;
        group_labels(shuffled_indices(start_index:end_index)) = i;
        start_index = end_index + 1;
    end

end

