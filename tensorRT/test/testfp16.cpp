#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "sys/timer.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

extern "C" 
{
void init(void* handle[1],
		  const char* deployFile,
		  const char* modelFile,				 
		  unsigned int maxBatchSize);

void caffeToGIEModel_(const char* deployFile,				
					  const char* modelFile,				
					  unsigned int maxBatchSize,					
					  std::ostream& gieModelStream);	

}

// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

void init(void* handle[1],
		  const char* deployFile,
		  const char* modelFile,				 
		  unsigned int maxBatchSize)	   
{
	// create a GIE model from the caffe model and serialize it to a stream
	std::stringstream gieModelStream;
	caffeToGIEModel_(deployFile, modelFile, maxBatchSize, gieModelStream);

	// deserialize the engine 
	gieModelStream.seekg(0, gieModelStream.beg);

	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream);
	IExecutionContext *context = engine->createExecutionContext();

	handle[1] = context;
}

void caffeToGIEModel_(const char* deployFile,				
					  const char* modelFile,				
					  unsigned int maxBatchSize,					
					  std::ostream& gieModelStream)				
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
    
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile,
															  modelFile,
															  *network,
															  DataType::kFLOAT);

	// specify which tensors are outputs
	printf("init finished");
	network->markOutput(*blobNameToTensor->find("prob"));
//	network->markOutput(*blobNameToTensor->find("bboxes"));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void loadimg(const char * inputpath,std::vector<cv::Mat> *input_channels)
{
	cv::Size input_geometry_(224,224);
    int m_num_channels =3;
    cv::Mat mean_;

//	cv::Mat temp = cv::imread(m_imgList[i*batchSizeNow+j], 1);
    cv::Mat sample =  cv::imread(inputpath, 1);
	
	cv::Scalar channel_mean(120.41,127.01,139.47);
    mean_ = cv::Mat(input_geometry_, CV_32FC3, channel_mean);

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
    else
      sample_resized = sample;

    cv::Mat sample_float;
    if (m_num_channels == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

    if (mean_.empty()){
    /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
    	cv::split(sample_float, *input_channels);
    }
    else
    {
        cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);
    /* This operation will write the separate BGR planes directly to the
         * input layer of the network because it is wrapped by the cv::Mat
         * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
    }
}
  


void doInference(void* handle[1], float* input,int inputsize, float * output,int outputsize, int batchSize)
{
//	int im_h = THFloatTensor_size(input, 1);
//	int im_w = THFloatTensor_size(input, 2);
//	int mask_h = THFloatTensor_size(output_mask, 1);
//	int mask_w = THFloatTensor_size(output_mask, 2);
	
	IExecutionContext* context = (IExecutionContext *)(handle[1]);
	const ICudaEngine& engine = context->getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	//assert(engine.getNbBindings() == 3);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex("data"), 
		outputMaskIndex = engine.getBindingIndex("prob");
	//	outputBoxIndex = engine.getBindingIndex("bboxes");
		
	int inputSize  = 1 * inputsize * sizeof(float);
	int outSize = 1 * outputsize* sizeof(float);
//	int outBoxSize = 1 * 4 * mask_h * mask_w * sizeof(float);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
	CHECK(cudaMalloc(&buffers[outputMaskIndex], outSize));
//	CHECK(cudaMalloc(&buffers[outputBoxIndex], outBoxSize));
	
//	float *mInputCPU = THFloatTensor_data(input);
//	float *mOutMaskCPU = THFloatTensor_data(output_mask);
//	float *mOutBoxCPU = THFloatTensor_data(output_box);

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice, stream));
	context->enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputMaskIndex], outSize, cudaMemcpyDeviceToHost, stream));
//	CHECK(cudaMemcpyAsync(mOutBoxCPU, buffers[outputBoxIndex], outBoxSize, cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputMaskIndex]));
//	CHECK(cudaFree(buffers[outputBoxIndex]));
}


int main(){

    void* handle[1];

 //   init(handle,"/home/zzj/qniudev/testsample/resnet-deploy-tupu-2016-09-30.prototxt",
 //   "/home/zzj/qniudev/testsample/resnet-tupu-2016-09-30_iter_1040000.caffemodel",10);

	    init(handle,"/home/zzj/qniudev/testsample/mnist/lenet.prototxt",
    "/home/zzj/qniudev/testsample/mnist/lenet_iter_3000.caffemodel",10);
	std::vector<cv::Mat> input_channels;
	float *input= (float*)malloc(224*244*3*sizeof(float));
	float * tempinput =input;
	for (int i = 0; i < 3; ++i)
    {
      cv::Mat channel(224, 224, CV_32FC1, tempinput);
      input_channels.push_back(channel);
      tempinput += 224 * 224;
    }
	loadimg("test.jpg",&input_channels);
	printf("size = %d\n",input_channels.size());
	float output[3];	
	for (size_t i = 0; i < 22; i++)
	{
		printf("loc %d is = %f\n",i,input[i]);
	}
	
        timeval start,end;

        gettimeofday(&start,NULL);

 
    doInference(handle,input,224*224*3, output,3, 1);


        delay(10);

        gettimeofday(&end,NULL);

	for (size_t i = 0; i < 3; i++)
	{
		printf("loc %d is = %f\n",i,output[i]);
	}
	printf("lasttime  is %ld\n",1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000)

    return 0;

}
