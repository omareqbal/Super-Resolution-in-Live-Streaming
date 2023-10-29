function imgs_h_transfer = run_fast(seq_name, seg_id, quality_level, img_width, img_height, num_frames, keras_model_file)


enc_dir = fullfile(cd, '..', 'segments');


% ------------------------------------------------------------------------
% Configurations
% ------------------------------------------------------------------------
QP = 27; % Quantization level!


% Although this is left as a parameter, there are few components where only
% sr_ratio = 2 is supported, right now.
sr_ratio = 2; 

% ------------------------------------------------------------------------
% Call HEVC to encode the sequence, and get the syntax elements
% ------------------------------------------------------------------------
% This function specifies where the HEVC dataset folder is
enc_params = make_encoding_param('num_frames', num_frames, 'QP', QP);

% Call the encoding function
yuv_name = sprintf('%s_%d_%d', seq_name, quality_level, seg_id);


% Output arguments
yuv_recon_name = fullfile(enc_dir, [yuv_name, '_recon.yuv']);
binary_name = fullfile(enc_dir, [yuv_name, '.bin']);


% Information about the encoding!
enc_info = struct('yuv_recon_name', {yuv_recon_name}, ...
    'binary_name', {binary_name}, ...
    'img_width', {img_width}, ...
    'img_height', {img_height}, ...
    'dst_dir', {enc_dir}, ...
    'seq_alias', {yuv_name});

st = tic;
% Get the syntax elements
dec_info = get_dumped_information(enc_params, enc_info);
tmp = toc(st);
fprintf('Run Decoder and dump frame dependency information - %f\n',tmp);

st = tic;
[Y_low_res, intra_recon, inter_mc, res_all, inter_mask, other_info] = ...
    parse_all_saved_info(dec_info, num_frames);
tmp = toc(st);
fprintf('Parse frame dependency information - %f\n', tmp);


% % Data structure to hold all of the compressed information
hevc_info = struct('intra_recon', {intra_recon}, ...
    'inter_mc', {inter_mc}, ...
    'res_all', {res_all}, ...
    'inter_mask', {inter_mask}, ...
    'other_info', {other_info});


% ------------------------------------------------------------------------
% Frame-by-frame SR
% ------------------------------------------------------------------------

% We support multiple super-resolution methods to benchmark against,
% but in this released code, we only include SRCNN to benchmark
% against.

persistent keras_sr_model;
st = tic;

if isempty(keras_sr_model)
    Y_low_size = size(Y_low_res{1});
    keras_sr_model = importKerasNetwork(keras_model_file,'ImageInputSize',[sr_ratio*Y_low_size(1),sr_ratio*Y_low_size(2)]);
end
tmp = toc(st);
fprintf('Load Keras SRCNN model - %f\n', tmp);

% imgs_h_sr_keras = cell(1, num_frames);
% imgs_h_bicubic = cell(1, num_frames);

st1 = tic;
% for i = 1:num_frames
%     imgs_h_sr_keras{i} = keras_SRCNN(Y_low_res{i}, sr_ratio, keras_sr_model);
% end

imgs_h_sr_keras = keras_SRCNN(Y_low_res{1}, sr_ratio, keras_sr_model);
tmp = toc(st1);
fprintf('Apply Keras SRCNN on %d frames - %f\n', 1, tmp);


% st = tic;
% for i = 1:num_frames
%     imgs_h_sr_keras{i} = double(imgs_h_sr_keras{i});
%     imgs_h_bicubic{i} = imresize(Y_low_res{i}, sr_ratio, 'bicubic');
% end
% tmp = toc(st);
% fprintf('Apply Bicubic upscaling on %d frames - %f\n', num_frames, tmp);

% ------------------------------------------------------------------------
% FAST algorithms
% ------------------------------------------------------------------------
fprintf('Upsampling remaining frames\n')

st = tic;
[imgs_h_transfer_keras, ~] = hevc_transfer_sr(...
    imgs_h_sr_keras, num_frames, hevc_info);
tmp = toc(st);
fprintf('Transfer SR output to rest %d frames - %f\n', num_frames-1, tmp);

imgs_h_transfer = zeros(num_frames,sr_ratio*img_height,sr_ratio*img_width,'uint8');
for i = 1:num_frames
    imgs_h_transfer(i,:,:) = uint8(imgs_h_transfer_keras{i});

% for i = 1:num_frames
%     % --------------------------------------------------------------------
%     % Visualize the SR results
%     % --------------------------------------------------------------------
    
%     imwrite(uint8(imgs_h_transfer_keras{i}), sprintf('frames_%s/fast_%d.bmp', seq_name, i), 'BMP');
%     imwrite(uint8(imgs_h_sr_keras{i}), sprintf('frames_%s/sr_%d.bmp', seq_name, i), 'BMP');
    
% end

end