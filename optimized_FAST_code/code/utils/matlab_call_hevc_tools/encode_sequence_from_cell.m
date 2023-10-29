function enc_info = encode_sequence_from_cell(img_height, img_width, num_frames, seq_name, ...
    enc_params, enc_dir)

if ~isfield(enc_params, 'QP')
    enc_params.QP = 27;
end


yuv_name = sprintf('%s_%dx%d_%d', seq_name, img_width, img_height, ...
    num_frames);

if ~exist('enc_dir', 'var') || isempty(enc_dir)
    enc_dir = fullfile(cd, '..', 'temp_data', yuv_name);
end

if ~exist(enc_dir, 'dir')
    mkdir(enc_dir);
end


% Output arguments
yuv_recon_name = fullfile(enc_dir, [yuv_name, '_recon.yuv']);
binary_name = fullfile(enc_dir, [yuv_name, '.str']);


% Information about the encoding!
enc_info = struct('yuv_recon_name', {yuv_recon_name}, ...
    'binary_name', {binary_name}, ...
    'img_width', {img_width}, ...
    'img_height', {img_height}, ...
    'dst_dir', {enc_dir}, ...
    'seq_alias', {yuv_name});

end