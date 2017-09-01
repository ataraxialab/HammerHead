CC=g++
C_FLAGS=-g -Wall -std=c++11
TENSORRT_FLAG=-lnvinfer -lnvcaffe_parser
CUDA_FLAGS=-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
OPENCV_LIB=-lopencv_core -lopencv_highgui -lopencv_imgproc
TAGET=testTensorrt

all:
        $(CC) $(C_FLAGS) $(CUDA_FLAGS) -c testfp16.cpp -o testfp16.o
        $(CC) $(C_FLAGS) testfp16.o -o $(TAGET) $(TENSORRT_FLAG) $(CUDA_FLAGS) $(OPENCV_LIB)


