import matlab.engine
import requests
import numpy as np
import time
import cv2

def get_seg(baseurl, quality, seg_id):
    url = baseurl + f'segments-{quality}/chunk-{seg_id}.bin'
        
    resp = requests.get(url)
    time.sleep(1)
    with open('../segments/chunk_{}_{}.bin'.format(quality, seg_id),'wb') as f:
        f.write(resp.content)


print('\n\n\nCall Fast 1\n')
# eng = matlab.engine.start_matlab()
eng = matlab.engine.connect_matlab()
# eng.parpool()
eng.gcp()       # Start parallel pool, if not exists

eng.addpath(eng.genpath(eng.fullfile(eng.cd(), 'utils')))
eng.addpath(eng.genpath(eng.fullfile(eng.cd(), 'fast_sr')))

quality = 2

num_frames = 240
frame_width = 320
frame_height = 176

seg_start = 1
seg_end = 20

baseurl = 'http://10.5.20.130:8000/'

total_stall = 0
num_stalls = 0

for i in range(seg_start, seg_end+1):
    print('\n\n\n---------------Segment {}-------------\n\n\n'.format(i))
    get_seg(baseurl, quality, i)
    img = eng.run_fast('chunk', i, quality, frame_width, frame_height, num_frames, 'keras_srcnn2.h5')
    frames = np.array(img._data).reshape(img.size, order='F')
    print(frames.shape)
    
    if(i > 1):
        en = time.time()
        stall = en - st
        total_stall += stall
        num_stalls += 1
        print('\n------Stall = {:.2f}------\n'.format(stall))
    
        
    for j in range(len(frames)):
    #     cv2.imwrite('frames_chunk_opencv/frame{}.png'.format(j), frames[j])
        cv2.imshow('video',frames[j])
        if(cv2.waitKey(34) & 0xFF == ord('q')):
            break

    st = time.time()

cv2.destroyAllWindows()

print('\n----Avg Stall = {:.2f}----\n'.format(total_stall/num_stalls))


# eng.delete(eng.gcp('nocreate'),nargout=0)

eng.quit()
