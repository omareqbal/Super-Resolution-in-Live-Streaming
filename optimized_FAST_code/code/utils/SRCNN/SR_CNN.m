function imh = SR_CNN(iml, ratio, model)

switch ratio
    case 2
        sr_model = model.x2_model;
    case 3
        sr_model = model.x3_model;
    case 4
        sr_model = model.x4_model;
    otherwise
        error('Unsupported upsampling ratio: %f', ratio);
end

st = tic;
iml = single(iml) / 255;
imh_bicubic = imresize(iml, ratio, 'bicubic');
tmp = toc(st);
fprintf('Upsample by bicubic interpolation - %f\n', tmp);

st = tic;
imh_sr = SRCNN(sr_model, imh_bicubic);
tmp = toc(st);
fprintf('Model Inference - %f\n', tmp);

st = tic;
border = ratio;
imh = imh_bicubic;
imh((1 + border):(end - border), (1 + border):(end - border)) = ...
    imh_sr((1 + border):(end - border), (1 + border):(end - border));

imh = uint8(255 * imh);
tmp = toc(st);
fprintf('Remove border - %f\n', tmp);

end