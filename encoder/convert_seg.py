import os
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir')
parser.add_argument('--quality', type = int)

args = parser.parse_args()

vid_dir = args.data_dir
ql = args.quality

os.makedirs(f'segments-{ql}',exist_ok=True)

hts = [180,180,270,360,360,432,576,720]
wts = [320,320,480,640,640,768,1024,1280]

seg_start = 1
seg_end = 20

num_frames = 240

for i in range(seg_start, seg_end+1):
    ht = hts[ql-2]
    wt = wts[ql-2]

    if(ht%16 != 0):
        ht -= ht%16
    if(wt%16 != 0):
        wt -= wt%16

    cmd = f'cat {vid_dir}/init-stream{ql}.m4s {vid_dir}/chunk-stream{ql}-{i:05d}.m4s > segments-{ql}/chunk-{i}_0.mp4'
    os.system(cmd)

    cmd = f'ffmpeg -i segments-{ql}/chunk-{i}_0.mp4 -filter:v "crop={wt}:{ht}:0:0" -c:a copy segments-{ql}/chunk-{i}_1.mp4'
    os.system(cmd)

    cmd = f'ffmpeg -i segments-{ql}/chunk-{i}_1.mp4 -c:v rawvideo -pix_fmt yuv420p segments-{ql}/chunk-{i}_1.yuv'
    os.system(cmd)

    cmd = f'./TAppEncoderStatic -c encoder_lowdelay_P_main.cfg -c bbb.cfg -i segments-{ql}/chunk-{i}_1.yuv -o segments-{ql}/chunk-{i}_1_recon.yuv -b segments-{ql}/chunk-{i}.bin --FramesToBeEncoded={num_frames} --SourceWidth={wt} --QP=27 --SourceHeight={ht}'
    os.system(cmd)

    cmd = f'rm segments-{ql}/chunk-{i}_*.mp4 segments-{ql}/chunk-{i}_*.yuv'
    os.system(cmd)
