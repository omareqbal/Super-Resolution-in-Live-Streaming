import skvideo.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

avg_p = 0
avg_s = 0
SR_frames = skvideo.io.vread('video_SR.mp4')
HR_frames = skvideo.io.vread('video_HR.mp4')
num_frames = SR_frames.shape[0]
print(num_frames)
for i in range(num_frames):
    psnr = peak_signal_noise_ratio(HR_frames[i], SR_frames[i])
    ssim = structural_similarity(HR_frames[i], SR_frames[i], multichannel=True, gaussian_weights=True)
    avg_p += psnr
    avg_s += ssim
    print(i, psnr, ssim)
print(psnr, ssim)
print('average',avg_p/num_frames, avg_s/num_frames)
