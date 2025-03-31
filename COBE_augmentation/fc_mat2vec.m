function vec = fc2vec(FCs, n)
    len = n*(n-1)/2;
    vec = zeros(size(FCs,3),len);
    for idx = 1:size(FCs,3)
        fc = squareform(tril(FCs(:,:,idx), -1));
        vec(idx,:) = fc;
    end
end