import argparse
import sys
import scipy
import os
import skvideo.io
from PIL import Image
import torch
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from skimage import io, transform
from model import ModelFactory
from torch.autograd import Variable
import time
description='Video Super Resolution pytorch implementation'

def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


parser = argparse.ArgumentParser(description=description)

parser.add_argument('-m', '--model', metavar='M', type=str, default='TDAN',
                    help='network architecture.')
parser.add_argument('-s', '--scale', metavar='S', type=int, default=4, 
                    help='interpolation scale. Default 4')
parser.add_argument('-t', '--test-set', metavar='NAME', type=str, default='/home/cxu-serve/u1/ytian21/project/video_retoration/TDAN-VSR/data/Vid4',
                    help='dataset for testing.')
parser.add_argument('-mp', '--model-path', metavar='MP', type=str, default='model',
                    help='model path.')
parser.add_argument('-sp', '--save-path', metavar='SP', type=str, default='res',
                    help='saving directory path.')
args = parser.parse_args()


frames = skvideo.io.vread('video.mp4')
print('Done reading frames')
print(frames.shape)

model_factory = ModelFactory()
model = model_factory.create_model(args.model)
model_path = os.path.join(args.model_path, 'model.pt')
if not os.path.exists(model_path):
    raise Exception('Cannot find %s.' %model_path)
model = torch.load(model_path)
model.eval()
path = args.save_path
if not os.path.exists(path):
            os.makedirs(path)


num, row, col, ch = frames.shape

frames_lr = np.zeros((5, int(row), int(col), ch))
outputs = [] 
for j in range(num):
    for k in range(j-2, j + 3):
        idx = k-j+2
        if k < 0:
            k = -k
        if k >= num:
            k = num - 3
        frames_lr[idx, :, :, :] = frames[k]
    start = time.time()
    frames_lr = frames_lr/255.0 - 0.5
    lr = torch.from_numpy(frames_lr).float().permute(0, 3, 1, 2)
    lr = Variable(lr.cuda()).unsqueeze(0).contiguous()
    output, _ = model(lr)
    #output = forward_x8(lr, model)
    output = (output.data + 0.5)*255
    output = quantize(output, 255)
    output = output.squeeze(dim=0)
    output = np.around(output.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
    outputs.append(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    elapsed_time = time.time() - start
    print(elapsed_time)
print('Done generating frames')

out = cv2.VideoWriter('video2.mp4', cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (1280,  720))
for j in range(num):
  out.write(outputs[j])
out.release()

print('Done writing video')

#st_time=$(ffprobe -v error -show_entries format=start_time -of default=noprint_wrappers=1:nokey=1 video.mp4)
#ffmpeg -i video2.mp4 -copyts -output_ts_offset $st_time output.mp4

#ffmpeg -i video_SR.mp4 -vcodec h264 -profile:v baseline -movflags frag_keyframe+empty_moov+default_base_moof -output_ts_offset 16 out_SR.mp4
       
