#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <unordered_map>

#include "BatchStream.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "common.h"
#include "argsParser.h"

#include "opencv2/opencv.hpp"
#pragma comment(lib, "opencv_core410.lib")
#pragma comment(lib, "opencv_highgui410.lib")
#pragma comment(lib, "opencv_imgcodecs410.lib")
#pragma comment(lib, "opencv_imgproc410.lib")
#pragma comment(lib, "opencv_video410.lib")
#pragma comment(lib, "opencv_videoio410.lib")

#include <chrono>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using std::vector;

const std::string gSampleName = "TensorRT.sample_ssd";
static samplesCommon::Args gArgs;

// Network details
const char* gNetworkName = "ssd";       // Network name
static const int kINPUT_C = 3;          // Input image channels
static const int kINPUT_H = 300;        // Input image height
static const int kINPUT_W = 300;        // Input image width
static const int kOUTPUT_CLS_SIZE = 21; // Number of classes
static const int kKEEP_TOPK = 200;      // Number of total bboxes to be kept per image after NMS step. It is same as detection_output_param.keep_top_k in prototxt file

enum MODE
{
	kFP32,
	kFP16,
	kINT8,
	kUNKNOWN
};

struct Param
{
	MODE modelType{ MODE::kFP32 }; // Default run FP32 precision
} params;

std::ostream& operator<<(std::ostream& o, MODE dt)
{
	switch (dt)
	{
	case kFP32: o << "FP32"; break;
	case kFP16: o << "FP16"; break;
	case kINT8: o << "INT8"; break;
	case kUNKNOWN: o << "UNKNOWN"; break;
	}
	return o;
}

// Data directory
static const std::vector<std::string> kDIRECTORIES{ "data/samples/ssd/", "data/ssd/", "data/int8_samples/ssd/" };

const std::string gCLASSES[kOUTPUT_CLS_SIZE]{ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" }; // List of class labels

static const char* kINPUT_BLOB_NAME = "data";            // Input blob name
static const char* kOUTPUT_BLOB_NAME0 = "detection_out"; // Output blob name
static const char* kOUTPUT_BLOB_NAME1 = "keep_count";    // Output blob name

// INT8 calibration variables
static const int kCAL_BATCH_SIZE = 1;   // Batch size
static const int kFIRST_CAL_BATCH = 0;  // First batch
static const int kNB_CAL_BATCHES = 100; // Number of batches

#define CalibrationMode 1 //Set to '0' for Legacy calibrator and any other value for Entropy calibrator 2

// Visualization
const float kVISUAL_THRESHOLD = 0.8f;

class Int8LegacyCalibrator : public nvinfer1::IInt8LegacyCalibrator
{
public:
	Int8LegacyCalibrator(BatchStream& stream, int firstBatch, double cutoff, double quantile, const char* networkName, bool readCache = true)
		: mStream(stream)
		, mFirstBatch(firstBatch)
		, mReadCache(readCache)
		, mNetworkName(networkName)
	{
		nvinfer1::Dims dims = mStream.getDims();
		mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
		CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
		reset(cutoff, quantile);
	}

	virtual ~Int8LegacyCalibrator()
	{
		CHECK(cudaFree(mDeviceInput));
	}

	int getBatchSize() const override { return mStream.getBatchSize(); }
	double getQuantile() const override { return mQuantile; }
	double getRegressionCutoff() const override { return mCutoff; }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		if (!mStream.next())
			return false;

		CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
		bindings[0] = mDeviceInput;
		return true;
	}

	const void* readCalibrationCache(size_t& length) override
	{
		mCalibrationCache.clear();
		std::ifstream input(calibrationTableName(), std::ios::binary);
		input >> std::noskipws;

		if (mReadCache && input.good())
		{
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
		}

		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output(calibrationTableName(), std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

	const void* readHistogramCache(size_t& length) override
	{
		length = mHistogramCache.size();
		return length ? &mHistogramCache[0] : nullptr;
	}

	void writeHistogramCache(const void* cache, size_t length) override
	{
		mHistogramCache.clear();
		std::copy_n(reinterpret_cast<const char*>(cache), length, std::back_inserter(mHistogramCache));
	}

	void reset(double cutoff, double quantile)
	{
		mCutoff = cutoff;
		mQuantile = quantile;
		mStream.reset(mFirstBatch);
	}

private:
	std::string calibrationTableName()
	{
		assert(mNetworkName != nullptr);
		return std::string("CalibrationTable") + mNetworkName;
	}
	BatchStream mStream;
	int mFirstBatch;
	double mCutoff, mQuantile;
	bool mReadCache{ true };
	const char* mNetworkName;
	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache, mHistogramCache;
};

class Int8EntropyCalibrator2 : public IInt8EntropyCalibrator2
{
public:
	Int8EntropyCalibrator2(BatchStream& stream, int firstBatch, bool readCache = true)
		: mStream(stream)
		, mReadCache(readCache)
	{
		nvinfer1::Dims dims = mStream.getDims();
		mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
		CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
		mStream.reset(firstBatch);
	}

	virtual ~Int8EntropyCalibrator2()
	{
		CHECK(cudaFree(mDeviceInput));
	}

	int getBatchSize() const override { return mStream.getBatchSize(); }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		if (!mStream.next())
		{
			return false;
		}
		CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
		assert(!strcmp(names[0], kINPUT_BLOB_NAME));
		bindings[0] = mDeviceInput;
		return true;
	}

	const void* readCalibrationCache(size_t& length) override
	{
		mCalibrationCache.clear();
		std::ifstream input(calibrationTableName(), std::ios::binary);
		input >> std::noskipws;
		if (mReadCache && input.good())
		{
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
		}
		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	virtual void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output(calibrationTableName(), std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

private:
	static std::string calibrationTableName()
	{
		assert(gNetworkName);
		return std::string("CalibrationTable") + gNetworkName;
	}
	BatchStream mStream;
	size_t mInputCount;
	bool mReadCache{ true };
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache;
};

std::string locateFile(const std::string& input)
{
	return locateFile(input, kDIRECTORIES);
}

void caffeToTRTModel(const std::string& deployFile,           // Name for caffe prototxt
	const std::string& modelFile,            // Name for model
	const std::vector<std::string>& outputs, // Network outputs
	unsigned int maxBatchSize,               // Batch size - NB must be at least as large as the batch we want to run with)
	MODE mode,                               // Precision mode
	IHostMemory** trtModelStream)            // Output stream for the TensorRT model
{
	// Create the builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	assert(builder != nullptr);

	// Parse the caffe model to populate the network, then set the outputs
	INetworkDefinition * network = builder->createNetwork();
	ICaffeParser * parser = createCaffeParser();
	DataType dataType = DataType::kFLOAT;
	if (mode == kFP16)
		dataType = DataType::kHALF;
	gLogInfo << "Begin parsing model..." << std::endl;
	gLogInfo << mode << " mode running..." << std::endl;

	const IBlobNameToTensor * blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
		locateFile(modelFile).c_str(),
		*network,
		dataType);
	gLogInfo << "End parsing model..." << std::endl;

	// Specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(36 << 20);
	builder->setFp16Mode(gArgs.runInFp16);
	builder->setInt8Mode(gArgs.runInInt8);
	builder->allowGPUFallback(true);
	samplesCommon::enableDLA(builder, gArgs.useDLACore);

	// Calibrator life time needs to last until after the engine is built.
	std::unique_ptr<IInt8Calibrator> calibrator;

	ICudaEngine * engine;
	if (mode == kINT8)
	{
#if CalibrationMode == 0
		assert(args.useDLACore != -1 && "Legacy calibration mode not supported with DLA.");
		gLogInfo << "Using Legacy Calibrator" << std::endl;
		BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, "./batches/batch_calibration", kDIRECTORIES);
		calibrator.reset(new Int8LegacyCalibrator(calibrationStream, 0, kCUTOFF, kQUANTILE, gNetworkName, true));
#else
		gLogInfo << "Using Entropy Calibrator 2" << std::endl;
		BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, "./batches/batch_calibration", kDIRECTORIES);
		calibrator.reset(new Int8EntropyCalibrator2(calibrationStream, kFIRST_CAL_BATCH));
#endif
		builder->setInt8Mode(true);
		builder->setInt8Calibrator(calibrator.get());
	}
	else
	{
		builder->setFp16Mode(mode == kFP16);
	}
	gLogInfo << "Begin building engine..." << std::endl;
	engine = builder->buildCudaEngine(*network);
	assert(engine);
	gLogInfo << "End building engine..." << std::endl;

	// Once the engine is built. Its safe to destroy the calibrator.
	calibrator.reset();

	// We don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// Serialize the engine, then close everything down
	(*trtModelStream) = engine->serialize();

	engine->destroy();
	builder->destroy();
}

void doInference(IExecutionContext & context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 1 input and 2 output.
	assert(engine.getNbBindings() == 3);
	void* buffers[3];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(kINPUT_BLOB_NAME),
		outputIndex0 = engine.getBindingIndex(kOUTPUT_BLOB_NAME0),
		outputIndex1 = engine.getBindingIndex(kOUTPUT_BLOB_NAME1);

	// Create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * kINPUT_C * kINPUT_H * kINPUT_W * sizeof(float))); // Data
	CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * kKEEP_TOPK * 7 * sizeof(float)));               // Detection_out
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * sizeof(int)));                                  // KeepCount (BBoxs left for each batch)

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * kINPUT_C * kINPUT_H * kINPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * kKEEP_TOPK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex0]));
	CHECK(cudaFree(buffers[outputIndex1]));
}

void printHelp(const char* name)
{
	std::cout << "Usage: " << name << "\n"
		<< "Optional Parameters:\n"
		<< "  -h, --help        Display help information.\n"
		<< "  --useDLACore=N    Specify the DLA engine to run on.\n"
		<< "  --fp16            Specify to run in fp16 mode.\n"
		<< "  --int8            Specify to run in int8 mode." << std::endl;
}

int main(int argc, char** argv)
{
	bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
	if (gArgs.help || !argsOK)
	{
		printHelp(argv[0]);
		return argsOK ? EXIT_SUCCESS : EXIT_FAILURE;
	}

	params.modelType = kFP32;
	if (gArgs.runInFp16)
	{
		params.modelType = kFP16;
	}
	else if (gArgs.runInInt8)
	{
		params.modelType = kINT8;
	}


	auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

	gLogger.reportTestStart(sampleTest);

	initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

	IHostMemory* trtModelStream{ nullptr };
	// Create a TensorRT model from the caffe model and serialize it to a stream

	const int N = 1; // Batch size

	caffeToTRTModel("ssd.prototxt",
		"VGG_VOC0712_SSD_300x300_iter_120000.caffemodel",
		std::vector<std::string>{kOUTPUT_BLOB_NAME0, kOUTPUT_BLOB_NAME1},
		N, params.modelType, & trtModelStream);

	gLogInfo << "*** deserializing" << std::endl;
	IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
	// Deserialize the engine
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
	assert(engine != nullptr);
	IExecutionContext * context = engine->createExecutionContext();
	assert(context != nullptr);
	trtModelStream->destroy();

	// Host memory for outputs
	float* detectionOut = new float[N * kKEEP_TOPK * 7];
	int* keepCount = new int[N];


	// Open Video Capture
	cv::VideoCapture cap("d:\\temp\\video.mp4");
	auto round = [](float x) -> int { return int(std::floor(x + 0.5f)); };
	auto adjust = [](float x, int d) -> float {
		float p = powf(10.0f, (float)d);
		return std::floor(x * p + 0.5f) / p;
	};

	cv::namedWindow("input_data", cv::WINDOW_NORMAL);
	cv::namedWindow("video_frame", cv::WINDOW_NORMAL);

	bool is_playing = true;
	int current_frame = 0;
	int last_frame = -1;

	int trackbar_value = kVISUAL_THRESHOLD * 100;

	cv::Mat video_frame;
	cv::Mat video_cache;
	int W = 300;
	int H = 300;

	while (true)
	{
		auto start = std::chrono::system_clock::now();

		// Read Video Frame
		if (is_playing)
		{
			cap >> video_frame;
			video_cache = video_frame.clone();
			current_frame++;
		}
		else
		{
			if (last_frame != current_frame)
			{
				cap.set(cv::CAP_PROP_POS_FRAMES, current_frame);
				cap.read(video_frame);
				video_cache = video_frame.clone();
			}
			else
			{
				video_frame = video_cache.clone();
			}
		}

		last_frame = current_frame;

		// Convert to 300x300
		cv::resize(video_frame, video_frame, cv::Size(W, H));

		// Create Input data
		cv::Mat frame_f;
		video_frame.convertTo(frame_f, CV_32FC3);
		auto pixelMean = cv::mean(frame_f);

		int	volChl = kINPUT_H * kINPUT_W;
		float* data = new float[kINPUT_C * kINPUT_H * kINPUT_W];
		for (int c = 0; c < kINPUT_C; ++c)
		{
			for (unsigned j = 0, volChl = kINPUT_H * kINPUT_W; j < volChl; ++j)
			{
				float d = (float)video_frame.data[j * kINPUT_C + 2 - c];
				data[c * volChl + j] = d - pixelMean[c];
			}
		}
		cv::Mat input_data(kINPUT_H, kINPUT_W, CV_32FC3, data);

		// Run inference
		doInference(*context, reinterpret_cast<float*>(input_data.data), detectionOut, keepCount, N);

		int p = 0;
		for (int i = 0; i < keepCount[p]; ++i)
		{
			float* det = detectionOut + (p * kKEEP_TOPK + i) * 7;
			if (det[2] < trackbar_value * 0.01)
				continue;
			assert((int)det[1] < kOUTPUT_CLS_SIZE);

			auto label = gCLASSES[(int)det[1]];
			if (label == "person")
			{
				samplesCommon::BBox bbox = { det[3] * kINPUT_W, det[4] * kINPUT_H, det[5] * kINPUT_W, det[6] * kINPUT_H };

				int x1 = std::min(std::max(0, round(int(bbox.x1))), W - 1);
				int x2 = std::min(std::max(0, round(int(bbox.x2))), W - 1);
				int y1 = std::min(std::max(0, round(int(bbox.y1))), H - 1);
				int y2 = std::min(std::max(0, round(int(bbox.y2))), H - 1);

				cv::Rect det_rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
				cv::rectangle(input_data, det_rect, { 0,0,255 }, 1);
				cv::rectangle(video_frame, det_rect, { 0,0,255 }, 1);

				std::stringstream ss;
				ss << label << ":" << adjust(det[2], 2);

				cv::putText(input_data, ss.str(), cv::Point(x1, y1 + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
				cv::putText(video_frame, ss.str(), cv::Point(x1, y1 + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

				//std::cout << label << ":" << det[2] << "," << x1 << "," << y1 << "," << x2 << "," << y2 << std::endl;
			}
		}

		cv::createTrackbar("x0.01", "input_data", &trackbar_value, 100);

		// FPS
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
		auto fps = 1000 / elapsed;
		std::stringstream ss_fps;
		ss_fps << "FPS:" << fps;
		cv::putText(input_data, ss_fps.str(), cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
		cv::putText(video_frame, ss_fps.str(), cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		cv::imshow("input_data", input_data);
		cv::imshow("video_frame", video_frame);

		delete[] data;

		auto wk = cv::waitKey(1);
		if (wk == 'q')
			break;
		else if (wk == 32)
		{
			is_playing = !is_playing;
		}
		else if (wk == 'z')
		{
			current_frame--;
		}
		else if (wk == 'x')
		{
			current_frame++;
		}
		else if (wk == 'a')
		{
			current_frame -= 100;
		}
		else if (wk == 's')
		{
			current_frame += 100;
		}
	}

	cap.release();

	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	delete[] detectionOut;
	delete[] keepCount;

	// Note: Once you call shutdownProtobufLibrary, you cannot use the parsers anymore.
	shutdownProtobufLibrary();

	//return gLogger.reportTest(sampleTest, pass);
}
