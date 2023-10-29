import matlab.engine
import requests
import numpy as np
import time
import cv2
import multiprocessing as mp
import random
import os
import argparse


def get_seg(baseurl, ql, seg_id):
    url = baseurl + f'segments-{ql}/chunk-{seg_id}.bin'
        
    resp = requests.get(url)
    time.sleep(1)
    with open(f'../segments/chunk_{ql}_{seg_id}.bin','wb') as f:
        f.write(resp.content)


def select_quality(ql_min, ql_max):
    ql = random.randint(ql_min, ql_max)
    return ql

def fetch_segments(baseurl, seg_start, seg_end, seg_fetched, ql_min, ql_max, ql_q):

    for i in range(seg_start, seg_end+1):
        ql = select_quality(ql_min, ql_max)
        print(f'\n\n\n-------------Fetching Segment {i}-------------\n\n\n')
        get_seg(baseurl, ql, i)

        ql_q.put(ql)
        seg_fetched.value += 1


def get_frames_from_bin(ql, seg_id):
    # convert .bin to .mp4
    fname = f'../segments/chunk_{ql}_{seg_id}'
    i_fname = f'{fname}.bin'
    o_fname = f'{fname}.mp4'
    cmd = f'ffmpeg -i {i_fname} -c copy {o_fname} -y >/dev/null 2>&1' # overwrite if exists and suppress output
    
    os.system(cmd)

    frames = []
    cap = cv2.VideoCapture(o_fname)
    while(True):
        ret, frame = cap.read()
        if(ret):
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break
    cap.release()
    
    return frames


def sr_segments(num_frames, frame_width_all, frame_height_all, seg_start, seg_end, seg_fetched, seg_sr, ql_min, ql_max, ql_q, frames_q):

    eng = matlab.engine.connect_matlab()

    eng.gcp() 

    eng.addpath(eng.genpath(eng.fullfile(eng.cd(), 'utils')))
    eng.addpath(eng.genpath(eng.fullfile(eng.cd(), 'fast_sr')))

    for i in range(seg_start, seg_end+1):
        while(seg_fetched.value < i):
            # print(seg_fetched.value, seg_sr.value)
            time.sleep(1)
        
        ql = ql_q.get()
        print(f'\n\n\n-------------SR Segment {i}      Quality {ql}-------------\n\n\n')
        # if quality = 2 or 3, then apply SR, else directly read the frames
        if(ql <= 3):
            frame_width = frame_width_all[ql - ql_min]
            frame_height = frame_height_all[ql - ql_min]

            img = eng.run_fast('chunk', i, ql, frame_width, frame_height, num_frames, 'keras_srcnn2.h5')
            frames = np.array(img._data).reshape(img.size, order='F')
        
        else:
            frames = get_frames_from_bin(ql, i)
        
        frames_q.put(frames)
        seg_sr.value += 1

    eng.quit()



def main(baseurl):

    print('\n\n\nCall Fast 4\n')
    os.makedirs('../segments',exist_ok=True)

    num_frames = 240

    ql_min = 2
    ql_max = 9

    frame_height_all = [176,176,256,352,352,432,576,720]
    frame_width_all = [320,320,480,640,640,768,1024,1280]

    seg_start = 1
    seg_end = 20


    seg_fetched = mp.Value('i',0)
    seg_sr = mp.Value('i',0)

    frames_q = mp.Queue()
    ql_q = mp.Queue()

    p1 = mp.Process(target=fetch_segments, args=(baseurl, seg_start, seg_end, seg_fetched, ql_min, ql_max, ql_q))
    p1.start()

    p2 = mp.Process(target=sr_segments, args=(num_frames, frame_width_all, frame_height_all, seg_start, seg_end, seg_fetched, seg_sr, ql_min, ql_max, ql_q, frames_q))
    p2.start()

    total_stall = 0
    num_stalls = 0

    for i in range(seg_start,seg_end+1):

        while(seg_sr.value < i):
            time.sleep(1)
        
        print(f'\n\n\n-------------Displaying Segment {i}-------------\n\n\n')
        frames = frames_q.get()

        if(i > seg_start):
            en = time.time()
            stall = en - st
            total_stall += stall
            num_stalls += 1
            print('\n------Stall = {:.2f}------\n'.format(stall))
        
        if(i == seg_start):
            cv2.namedWindow('video',cv2.WINDOW_NORMAL)

        for j in range(len(frames)):
            # cv2.imwrite(f'frames_all_opencv_copy/Segment{i}_Frame{j}.png',frames[j])
            cv2.imshow('video',frames[j])
            if(cv2.waitKey(34) & 0xFF == ord('q')):
                break

        st = time.time()

    cv2.destroyAllWindows()

    print('\n----Avg Stall = {:.2f}----\n'.format(total_stall/num_stalls))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server')

    args = parser.parse_args()
    main(args.server)