close all
clear;
seed = 1;
rng(seed);
nfold = 10;
p_value = 0.05;
outlier = num2str(0.1);
% eye = {'REO', 'REC'};
eye = {'REO'};
bands = {'ALPHA'};
n_comp = 1:5; % number of augmented sets


ser_file = 'D:\EMBARC_GNN_multimodal\preprocessed_FC\ser_all_fmri_rs_allconfound_full_continues.mat';
pla_file = 'D:\EMBARC_GNN_multimodal\preprocessed_FC\pla_all_fmri_rs_allconfound_full_continues.mat'; 
load(ser_file);
load(pla_file);  

%%%%%%%%%%% DATA INFO %%%%%%%%%%
sheet = readcell('D:/Lehigh University Dropbox/ENG-ZhangBICLab/BICLabData/EMBARC/ClinicalData/embarc clinical variables summary 022617 gbc with outcome_gfupdated_3-5-18_excel95.xls');
treat_index = find(strcmpi(sheet(1, :), 'Stage1TX'));
treatments = sheet(2:end, treat_index);

ser_rows = find(strcmpi(treatments, "SER")) + 1;
ser_subj = sheet(ser_rows, strcmpi(sheet(1, :), 'subj_ID'));
pla_rows = find(strcmpi(treatments, "PLA")) + 1;
pla_subj = sheet(pla_rows, strcmpi(sheet(1,:), "subj_ID"));

%%
%%%%%%%%%%%%%% fMRI %%%%%%%%%%%%%%

ser_hamd_week0 = ser_fmri.s_pcd_w0_hdrs;
pla_hamd_week0 = pla_fmri.p_pcd_w0_hdrs;
ser_hamd_week8 = ser_fmri.s_pcd_w8_hdrs;
pla_hamd_week8 = pla_fmri.p_pcd_w8_hdrs;
ser_hamd_diff_fmri = (ser_hamd_week0 - ser_hamd_week8)';
pla_hamd_diff_fmri = (pla_hamd_week0 - pla_hamd_week8)';

ser_fmri_ts1 = ser_fmri.s_im_fmri1(:,:,:,1);
pla_fmri_ts1 = pla_fmri.p_im_fmri1(:,:,:,1);
ser_fmri_ts2 = ser_fmri.s_im_fmri2(:,:,:,1);
pla_fmri_ts2 = pla_fmri.p_im_fmri2(:,:,:,1);
ser_subName_fmri1 = cellstr(ser_fmri.s_pcd_id1);
pla_subName_fmri1 = cellstr(pla_fmri.p_pcd_id1);
ser_subName_fmri2 = cellstr(ser_fmri.s_pcd_id2);
pla_subName_fmri2 = cellstr(pla_fmri.p_pcd_id2);

ser_subName_fmri = [ser_subName_fmri1;ser_subName_fmri2];
pla_subName_fmri = [pla_subName_fmri1;pla_subName_fmri2];

[~,ser_idx,~]=intersect(ser_subName_fmri1,ser_subName_fmri2,'stable');
[~,pla_idx,~]=intersect(pla_subName_fmri1,pla_subName_fmri2,'stable');

ser_target = [ser_hamd_diff_fmri; ser_hamd_diff_fmri(ser_idx)];
pla_target = [pla_hamd_diff_fmri; pla_hamd_diff_fmri(pla_idx)];

ser_fmri_ts = cat(3, ser_fmri_ts1, ser_fmri_ts2);
ser_pear_fmri = fc_mat2vec(ser_fmri_ts, 100);
ser_pear_fmri = atanh(ser_pear_fmri.').';
ser_pear_fmri = zscore(ser_pear_fmri')';

pla_fmri_ts = cat(3, pla_fmri_ts1, pla_fmri_ts2);
pla_pear_fmri = fc_mat2vec(pla_fmri_ts, 100);
pla_pear_fmri = atanh(pla_pear_fmri.').';
pla_pear_fmri = zscore(pla_pear_fmri')';

%%
for e=1:length(eye)
    for b = 1:length(bands)
        disp(['Eye: ', char(eye(e)), ', Band: ', char(bands(b))]);

        %%%%%%%%% EEG  %%%%%%%%%%%
        root = 'D:/Lehigh University Dropbox/ENG-ZhangBICLab/BICLabData/EMBARC/Resting_EEG/ExtractNetwork_LogFisher_EMBARC_Final_Baseline/Schaeffer100ROI_powenv_';
        file_name = sprintf('%s%s_MNE_%s_Baseline.mat', root, eye{e}, bands{b});
        load(file_name)

        [ser_subName_eeg, sub_ser_idx, ser_idx] = intersect(cellstr(ser_subj), cellstr(subjectID),'stable');
        [pla_subName_eeg, sub_pla_idx, pla_idx] = intersect(cellstr(pla_subj), cellstr(subjectID),'stable');
        
        ser_eeg_fc = ROIConn(:,:,ser_idx);
        pla_eeg_fc = ROIConn(:,:,pla_idx);
        
        % ser_hamd_diff_eeg = cell2mat(sheet(ser_rows, 3));
        % ser_hamd_diff_eeg = ser_hamd_diff_eeg(sub_ser_idx);
        % pla_hamd_diff_eeg = cell2mat(sheet(pla_rows, 3));
        % pla_hamd_diff_eeg = pla_hamd_diff_eeg(sub_pla_idx);
        
        ser_pear_eeg = fc_mat2vec(ser_eeg_fc, 100);
        % ser_feat = atanh(ser_feat.').';
        ser_pear_eeg = zscore(ser_pear_eeg.').';
        
        pla_pear_eeg = fc_mat2vec(pla_eeg_fc, 100);
        % pla_feat = atanh(pla_feat.').';
        pla_pear_eeg=zscore(pla_pear_eeg.').';

        % find subj with both fmri and EEG
        [ser_subName,IA, IB] = intersect(ser_subName_fmri, ser_subName_eeg);   % 130 subjs
        [pla_subName,IAA, IBB] = intersect(pla_subName_fmri, pla_subName_eeg); % 135 subjs
        % save('subj_ID.mat',"ser_subName","pla_subName")   
        
        ser_subName_idx = contains(ser_subName_fmri, ser_subName);
        pla_subName_idx = contains(pla_subName_fmri, pla_subName);

        ser_subName_fmri_band = ser_subName_fmri(ser_subName_idx);
        pla_subName_fmri_band = pla_subName_fmri(pla_subName_idx);
        ser_pear_fmri_band = ser_pear_fmri(ser_subName_idx,:);
        pla_pear_fmri_band = pla_pear_fmri(pla_subName_idx,:); 
        ser_hamd_diff = ser_target(ser_subName_idx); 
        pla_hamd_diff = pla_target(pla_subName_idx);

        ser_pear_eeg = ser_pear_eeg(IB,:);
        pla_pear_eeg = pla_pear_eeg(IBB,:);


        ser_indices = mycrossvalind(length(ser_subName), nfold);
        ser_aug = aug_func(ser_subName, ser_subName_fmri_band, ser_pear_fmri_band, ser_pear_eeg, ser_hamd_diff, pla_pear_fmri, pla_pear_eeg, ser_indices, n_comp);
        
        pla_indices = mycrossvalind(length(pla_subName), nfold);
        pla_aug = aug_func(pla_subName, pla_subName_fmri_band, pla_pear_fmri_band, pla_pear_eeg, pla_hamd_diff, ser_pear_fmri, ser_pear_eeg, pla_indices, n_comp);
        
        save_path = './Augmented_5times/';
        mkdir(save_path);
        save_name = sprintf('fMRI_EEG_%s_%s.mat', eye{e}, bands{b});
        save([save_path, save_name],"ser_aug","pla_aug")                   
    end
end
