function imh = keras_SRCNN(iml, ratio, model)

% tic
iml = single(iml) / 255;
imh_bicubic = imresize(iml, ratio, 'bicubic');
% tmp = toc;
% fprintf('Keras Upsample by bicubic interpolation - %f\n', tmp);

% tic
imh_sr = predict(model, imh_bicubic);
% tmp = toc;
% fprintf('Keras Model Inference - %f\n', tmp);

% tic
border = ratio;
imh = imh_bicubic;
imh((1 + border):(end - border), (1 + border):(end - border)) = ...
    imh_sr((1 + border):(end - border), (1 + border):(end - border));

imh = uint8(255 * imh);
% tmp = toc;
% fprintf('Keras Remove border - %f\n', tmp);

end