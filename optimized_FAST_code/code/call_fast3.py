import matlab.engine
import requests
import numpy as np
import time
import cv2
import multiprocessing as mp
import argparse
import os


def get_seg(baseurl, quality, seg_id):
    url = baseurl + f'segments-{quality}/chunk-{seg_id}.bin'
        
    resp = requests.get(url)
    time.sleep(1)
    with open('../segments/chunk_{}_{}.bin'.format(quality, seg_id),'wb') as f:
        f.write(resp.content)


def fetch_segments(baseurl, quality, seg_start, seg_end, seg_fetched):

    for i in range(seg_start, seg_end+1):
        print(f'\n\n\n-------------Fetching Segment {i}-------------\n\n\n')
        get_seg(baseurl, quality, i)
        seg_fetched.value += 1


def sr_segments(quality, num_frames, frame_width, frame_height, seg_start, seg_end, seg_fetched, seg_sr, frames_q):

    eng = matlab.engine.connect_matlab()

    eng.gcp() 

    eng.addpath(eng.genpath(eng.fullfile(eng.cd(), 'utils')))
    eng.addpath(eng.genpath(eng.fullfile(eng.cd(), 'fast_sr')))

    for i in range(seg_start, seg_end+1):
        while(seg_fetched.value < i):
            # print(seg_fetched.value, seg_sr.value)
            time.sleep(1)
        print(f'\n\n\n-------------SR Segment {i}-------------\n\n\n')
        img = eng.run_fast('chunk', i, quality, frame_width, frame_height, num_frames, 'keras_srcnn2.h5')
        frames = np.array(img._data).reshape(img.size, order='F')
        frames_q.put(frames)
        seg_sr.value += 1

    eng.quit()



def main(baseurl):

    print('\n\n\nCall Fast 3\n')
    os.makedirs('../segments',exist_ok=True)

    num_frames = 240
    frame_width = 320
    frame_height = 176

    quality = 2

    seg_start = 1
    seg_end = 20


    seg_fetched = mp.Value('i',0)
    seg_sr = mp.Value('i',0)

    frames_q = mp.Queue()

    p1 = mp.Process(target=fetch_segments, args=(baseurl, quality, seg_start, seg_end, seg_fetched))
    p1.start()

    p2 = mp.Process(target=sr_segments, args=(quality, num_frames, frame_width, frame_height, seg_start, seg_end, seg_fetched, seg_sr, frames_q))
    p2.start()

    total_stall = 0
    num_stalls = 0

    for i in range(seg_start,seg_end+1):

        while(seg_sr.value < i):
            time.sleep(1)
        
        print(f'\n\n\n-------------Displaying Segment {i}-------------\n\n\n')
        frames = frames_q.get()

        if(i > 1):
            en = time.time()
            stall = en - st
            total_stall += stall
            num_stalls += 1
            print('\n------Stall = {:.2f}------\n'.format(stall))
        

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