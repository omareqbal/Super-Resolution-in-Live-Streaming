import os
import torch
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from torch.autograd import Variable
import time
description='Video Super Resolution pytorch implementation'

def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


def get_frames(frames):
    num = len(frames)
    row, col, ch = frames[0].shape

    frames_lr = np.zeros((5, int(row), int(col), ch))
    outputs = [] 
    for j in range(num):
        for k in range(j-2, j + 3):
            idx = k-j+2
            if k < 0:
                k = -k
            if k >= num:
                k = num - 3
            frames_lr[idx, :, :, :] = cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB)
        start = time.time()
        frames_lr = frames_lr/255.0 - 0.5
        lr = torch.from_numpy(frames_lr).float().permute(0, 3, 1, 2)
        lr = Variable(lr.cuda()).unsqueeze(0).contiguous()
        output, _ = model(lr)
        output = (output.data + 0.5)*255
        output = quantize(output, 255)
        output = output.squeeze(dim=0)
        output = np.around(output.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
        outputs.append(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        elapsed_time = time.time() - start
        print(elapsed_time)
    return outputs


model_path = 'model/model.pt'

model = torch.load(model_path)
model.eval()