import matlab.engine
import requests
import numpy as np
import time
import cv2
import threading


seg_fetched = 0
seg_sr = 0

frames_all = []

def get_seg(baseurl, quality, seg_id):
    url = baseurl + f'segments-{quality}/chunk-{seg_id}.bin'
        
    resp = requests.get(url)
    time.sleep(1)
    with open('../segments/chunk_{}_{}.bin'.format(quality, seg_id),'wb') as f:
        f.write(resp.content)


def fetch_segments(baseurl, quality, seg_start, seg_end):
    global seg_fetched

    for i in range(seg_start, seg_end+1):
        print(f'\n\n\n-------------Fetching Segment {i}-------------\n\n\n')
        get_seg(baseurl, quality, i)
        seg_fetched += 1


def sr_segments(quality, num_frames, frame_width, frame_height, seg_start, seg_end):
    global seg_sr

    eng = matlab.engine.connect_matlab()

    eng.gcp() 

    eng.addpath(eng.genpath(eng.fullfile(eng.cd(), 'utils')))
    eng.addpath(eng.genpath(eng.fullfile(eng.cd(), 'fast_sr')))

    for i in range(seg_start, seg_end+1):
        while(seg_fetched < i):
            time.sleep(1)
        print(f'\n\n\n-------------SR Segment {i}-------------\n\n\n')
        img = eng.run_fast('chunk', i, quality, frame_width, frame_height, num_frames, 'keras_srcnn2.h5')
        frames = np.array(img._data).reshape(img.size,order='F')
        frames_all.append(frames)
        seg_sr += 1

    eng.quit()    


def main():

    print('\n\n\nCall Fast 2\n')

    num_frames = 240
    frame_width = 320
    frame_height = 176

    quality = 2

    seg_start = 1
    seg_end = 20

    baseurl = 'http://10.5.20.130:8000/'


    thread1 = threading.Thread(target=fetch_segments, args=(baseurl, quality, seg_start, seg_end))
    thread1.start()

    thread2 = threading.Thread(target=sr_segments, args=(quality, num_frames, frame_width, frame_height, seg_start, seg_end))
    thread2.start()

    total_stall = 0
    num_stalls = 0

    for i in range(seg_start,seg_end+1):

        while(seg_sr < i):
            time.sleep(1)
        
        print(f'\n\n\n-------------Displaying Segment {i}-------------\n\n\n')
        frames = frames_all[i-1]

        if(i > 1):
            en = time.time()
            stall = en - st
            total_stall += stall
            num_stalls += 1
            print('\n------Stall = {:.2f}------\n'.format(stall))
        
        for j in range(len(frames)):
            # cv2.imwrite(f'frames_all_opencv/Segment{i}_Frame{j}.png',frames[j])
            cv2.imshow('video',frames[j])
            if(cv2.waitKey(34) & 0xFF == ord('q')):
                break
            
        st = time.time()

    cv2.destroyAllWindows()

    print('\n----Avg Stall = {:.2f}----\n'.format(total_stall/num_stalls))



if __name__ == "__main__":
    main()