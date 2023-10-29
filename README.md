# Super Resolution in Live Streaming

M.Tech Thesis - https://drive.google.com/drive/folders/1LeLvSfcVMgWNNIA7eTR5DnAw9eM7oLeP?usp=sharing

### Dependencies
MATLAB -\
Parallel Computing Toolbox\
Deep Learning Toolbox\
Deep Learning Toolbox Importer for Tensorflow-Keras Models\
Statistics and Machine Learning Toolbox\
Image Processing Toolbox\
Symbolic Math Toolbox

Python - opencv, requests, numpy, MATLAB engine

To install MATLAB engine in Python, run the following in MATLAB prompt -
~~~
cd (fullfile(matlabroot,'extern','engines','python'))
system('python setup.py install')
~~~

### Evaluating TDAN
The folder evaluating_TDAN contains the code for TDAN taken from https://github.com/YapengTian/TDAN-VSR-CVPR-2020. It runs the TDAN model and computes metrics like PSNR and SSIM on it.


### Encoder
The encoder folder contains the code for encoding the segments to HEVC. The video segments used are from [DASHVideos](https://github.com/abhimp/DASHVideos). The encoder needs to be run for each of the quality levels 2-9.
~~~
cd encoder
python convert_seg.py --data_dir ../DASHVideos/bbb/media --quality 2
~~~
Start a python http server in encoder folder.
~~~
python -m http.server
~~~


### Player
The code for the player is in the optimized_FAST_code/code folder. The player is written in python and it uses the MATLAB code for FAST which is taken from https://www.mit.edu/~sze/fast.html. We have also modified the FAST code to use a keras SRCNN model which has much lesser inference time than the SRCNN model in MATLAB. The file call_fast3.py is the player which downloads only the quality level 2 segments and applies super resolution on all the segments. The file call_fast4.py is the player which randomly selects the quality to fetch and applies super resolution on the segments of quality levels 2 and 3. 
~~~
cd optimized_FAST_code/code
python call_fast3.py --server http://localhost:8000/

python call_fast4.py --server http://localhost:8000/
~~~
