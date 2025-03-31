function aug = aug_func(ser_subName, ser_subName_fmri, ser_pear_fmri, ser_pear_eeg,ser_hamd_diff,...
                                    pla_pear_fmri, pla_pear_eeg, indices, comps)

    unique_values = unique(indices);
    nfold = length(unique_values);

    gps = 10;    % divide into 10 groups
    for i = 1:nfold
        in_fold = num2str(i);
        subID_test=ser_subName(indices==unique_values(i)); subID_train=ser_subName(indices~=unique_values(i));
        
        test_fmri = contains(ser_subName_fmri,subID_test); 
        train_fmri = contains(ser_subName_fmri,subID_train);
        fmri_tr_x1 = ser_pear_fmri(train_fmri,:);
        fmri_tr_x2 = pla_pear_fmri;   % use pla data as supplement
        fmri_tr_x = [fmri_tr_x1; fmri_tr_x2];
        [fmri_tr_x,u1,s1] = normalize(fmri_tr_x);
        fmri_te_x = ser_pear_fmri(test_fmri,:);
        fmri_te_x = normalize(fmri_te_x,"center",u1,"scale",s1);

        test_eeg=contains(ser_subName,subID_test);
        train_eeg=contains(ser_subName,subID_train);
        eeg_tr_x1 = ser_pear_eeg(train_eeg,:);
        eeg_tr_x2 = pla_pear_eeg;
        eeg_tr_x = [eeg_tr_x1; eeg_tr_x2];
        [eeg_tr_x,u2,s2] = normalize(eeg_tr_x);
        eeg_te_x = ser_pear_eeg(test_eeg,:);
        eeg_te_x = normalize(eeg_te_x,"center",u2,"scale",s2);

        % COBE
        idd1=floor(linspace(0,size(fmri_tr_x,1),gps+1));
        idd1=diff(idd1);
        A=mat2cell(fmri_tr_x,idd1,size(fmri_tr_x,2));
        A=cellfun(@(c) c',A,'UniformOutput',false); %transpose

        idd2=floor(linspace(0,size(eeg_tr_x,1),gps+1));
        idd2=diff(idd2);
        B=mat2cell(eeg_tr_x,idd2,size(eeg_tr_x,2));
        B=cellfun(@(c) c',B,'UniformOutput',false); %transpose

        % original data + aug
        fmri_train_aug = zeros(length(comps)+1,size(fmri_tr_x1, 1),4950);
        fmri_train_aug(1,:,:) = fmri_tr_x1;
        fmri_test_aug = zeros(length(comps)+1,size(fmri_te_x, 1), 4950);
        fmri_test_aug(1,:,:) = fmri_te_x;
        eeg_train_aug = zeros(length(comps)+1,size(eeg_tr_x1, 1), 4950);
        eeg_train_aug(1,:,:) = eeg_tr_x1;
        eeg_test_aug = zeros(length(comps)+1,size(eeg_te_x, 1), 4950);
        eeg_test_aug(1,:,:) = eeg_te_x;

        for n = 1: length(comps)
            n_comm=comps(n);
            %%%%% fMRI
            [c,Q,~,~]=cobe_zy(A,n_comm);
            Q = cell2mat(Q);
            tr_x = [A{:}]-c*Q;
            tr_x = tr_x.';
            tr_x = tr_x(1:size(fmri_tr_x1,1),:);
            fmri_train_aug(n+1,:,:) = tr_x;

            fmri_te_x = fmri_te_x' - c*c'*fmri_te_x';
            fmri_te_x = fmri_te_x';
            fmri_test_aug(n+1,:,:) = fmri_te_x;
            %%%%% EEG
            [c,Q,~,~]=cobe_zy(B,n_comm);
            Q = cell2mat(Q);
            tr_x = [B{:}]-c*Q;
            tr_x = tr_x.';
            tr_x = tr_x(1:size(eeg_tr_x1,1),:);
            eeg_train_aug(n+1,:,:) = tr_x;

            eeg_te_x = eeg_te_x' - c*c'*eeg_te_x';
            eeg_te_x = eeg_te_x';
            eeg_test_aug(n+1,:,:) = eeg_te_x;
        end

        train_subName_fmri = ser_subName_fmri(train_fmri);
        test_subName_fmri = ser_subName_fmri(test_fmri);

        train_subName_eeg = ser_subName(train_eeg);
        test_subName_eeg = ser_subName(test_eeg);

        % [~, idx_test] = ismember(test_subName, test_subName_eeg);
        % [~, idx_train] = ismember(train_subName, train_subName_eeg);
        % train_X_eeg = eeg_train_aug(:,idx_train,:);
        % test_X_eeg = eeg_test_aug(:,idx_test,:);

        aug.(['fold', in_fold]).('train_subID_fmri') = train_subName_fmri;
        aug.(['fold', in_fold]).('test_subID_fmri') = test_subName_fmri;
        aug.(['fold', in_fold]).('train_subID_eeg') = train_subName_eeg;
        aug.(['fold', in_fold]).('test_subID_eeg') = test_subName_eeg;
        aug.(['fold', in_fold]).('train_target') = ser_hamd_diff(train_fmri);
        aug.(['fold', in_fold]).('test_target') = ser_hamd_diff(test_fmri);
        aug.(['fold', in_fold]).('train_X_fmri') = fmri_train_aug;
        aug.(['fold', in_fold]).('test_X_fmri') = fmri_test_aug;
        aug.(['fold', in_fold]).('train_X_eeg') = eeg_train_aug;
        aug.(['fold', in_fold]).('test_X_eeg') = eeg_test_aug;        
    end
end

