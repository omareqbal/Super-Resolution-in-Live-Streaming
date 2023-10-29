function [final_Y, intra_recon, inter_mc, recon_res, inter_mask, other_info] = ...
    parse_all_saved_info(dec_info, num_frames)

% ------------------------------------------------------------------------
% Parse the dumped.txt
% ------------------------------------------------------------------------
st1 = tic;
[PU_all, res_luma_all] = parse_dumped_coeff_multiple_poc(...
    dec_info.dump_txt_name, num_frames);
img_width = dec_info.enc_info.img_width;
img_height = dec_info.enc_info.img_height;
N_frames = length(PU_all);
tmp1 = toc(st1);
fprintf('Parse dump.txt - %f\n', tmp1);


% ------------------------------------------------------------------------
% Load the decoded sequence
% ------------------------------------------------------------------------
st1 = tic;
final_Y = load_Y_of_yuv(dec_info.enc_info.yuv_recon_name, img_width, ...
    img_height, N_frames);
tmp1 = toc(st1);
fprintf('Load sequence - %f\n', tmp1);
% ------------------------------------------------------------------------
% Reconstruct Residue
% ------------------------------------------------------------------------
st1 = tic;
recon_res = cell(1, N_frames);
% parfor f_idx = 1:N_frames
for f_idx = 1:N_frames
    recon_res{f_idx} = reconstruct_res(res_luma_all{f_idx}, ...
        [img_height, img_width]);
end
tmp1 = toc(st1);
fprintf('Reconstruct residue - %f\n', tmp1);
% ------------------------------------------------------------------------
% Reconstruct inter
% ------------------------------------------------------------------------
st1 = tic;
[inter_mc, inter_mask, mv_x_map, mv_y_map] = ...
    get_recon_inter(PU_all, final_Y);
tmp1 = toc(st1);
fprintf('Reconstruct inter - %f\n', tmp1);
% ------------------------------------------------------------------------
% Reconstruct intra
% ------------------------------------------------------------------------
st1 = tic;
intra_recon = cell(1, N_frames);
% parfor f_idx = 1:N_frames
for f_idx = 1:N_frames
    intra_recon{f_idx} = zeros(img_height, img_width);
    intra_mask_frame = inter_mask{f_idx} == 0;
    intra_recon{f_idx}(intra_mask_frame) = final_Y{f_idx}(intra_mask_frame);
end

% ------------------------------------------------------------------------
% Output Other Information
% ------------------------------------------------------------------------
other_info = struct('PU', {PU_all}, 'TU', {res_luma_all}, ...
    'mv_x', {mv_x_map}, 'mv_y', {mv_y_map});

tmp1 = toc(st1);
fprintf('Reconstruct intra - %f\n', tmp1);

end