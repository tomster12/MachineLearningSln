#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>

#include "Matrix.h"
#include "Utility.h"
#include "SupervisedNetwork.h"
#include "MNIST.h"
#include "ThreadPool.h"

void testBasic();
void testTime();
void testTimeThreaded();
void testBackprop();
void testMNIST();

int main()
{
	testMNIST();
	return 0;
}

void testBasic()
{
	// Create network, inputs, and run
	tbml::NeuralNetwork network = tbml::NeuralNetwork({ { 3, 3, 1 } }, { tbml::fn::Sigmoid(), tbml::fn::Sigmoid() });
	tbml::Matrix input = tbml::Matrix({ { 1.0f, -1.0f, 1.0f } });
	tbml::Matrix output = network.propogate(input);

	// Print values
	network.printLayers();
	input.printValues("Input:");
	output.printValues("Output:");
}

void testTime()
{
	// Create network and inputs
	tbml::NeuralNetwork network({ 8, 8, 8, 1 });
	tbml::Matrix input = tbml::Matrix({ { 1, 0, -1, 0.2f, 0.7f, -0.3f, -1, -1 } });
	size_t epoch = 10'000'000;

	// Number of epochs propogation timing
	// -----------
	// Release x86	1'000'000   ~1100ms
	// Release x86	1'000'000   ~600ms		Change to vector subscript from push_back
	// Release x86	3'000'000   ~1900ms
	// Release x86	3'000'000   ~1780ms		Update where icross initialises matrix
	// Release x86	3'000'000   ~1370ms		(Reverted) Make icross storage matrix static (cannot const or thread)
	// Release x86	10'000'000	~6000ms
	// Release x86	10'000'000	~7800ms		1D matrix
	// -----------
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	tbml::PropogateCache cache;
	for (size_t i = 0; i < epoch; i++) network.propogate(input, cache);
	std::cout << cache.neuronOutput[3](0, 0) << std::endl;
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

	// Print output
	network.printLayers();
	input.printValues("Input: ");
	std::cout << std::endl << "Epochs: " << epoch << std::endl;
	std::cout << "Time taken: " << us.count() / 1000.0f << "ms" << std::endl;
}

void testTimeThreaded()
{
	// Create network and inputs
	tbml::NeuralNetwork network(std::vector<size_t>({ 8, 8, 8, 1 }));
	tbml::Matrix input = tbml::Matrix({ { 1, 0, -1, 0.2f, 0.7f, -0.3f, -1, -1 } });
	size_t epoch = 50'000'000;

	// Number of epochs propogation timing
	// NOTE: If using threaded cross actually slows down
	// -----------
	// Release x86	50'000'000   ~6500ms
	// -----------
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	ThreadPool threadPool;
	std::vector<std::future<void>> results(threadPool.size());
	const size_t count = epoch / threadPool.size();
	for (size_t i = 0; i < threadPool.size(); i++)
	{
		results[i] = threadPool.enqueue([=]
		{
			for (size_t o = 0; o < count; o++) network.propogate(input);
		});
	}
	for (auto&& result : results) result.get();
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

	// Print output
	network.printLayers();
	input.printValues("Input: ");
	std::cout << std::endl << "Epochs: " << epoch << std::endl;
	std::cout << "Time taken: " << us.count() / 1000.0f << "ms" << std::endl;
}

void testBackprop()
{
	const float L = -1.0f, H = 1.0f;

	// Create network and setup training data
	tbml::SupervisedNetwork network({ 2, 2, 1 }, { tbml::fn::TanH(), tbml::fn::TanH() }, tbml::fn::SquareError());
	tbml::Matrix input = tbml::Matrix({
		{ L, L },
		{ L, H },
		{ H, L },
		{ H, H }
		});
	tbml::Matrix expected = tbml::Matrix({
		{ L },
		{ H },
		{ H },
		{ L }
		});

	// Print values and train
	input.printValues("Input:");
	expected.printValues("Expected:");
	network.propogate(input).printValues("Initial: ");
	network.train(input, expected, { -1, -1, 0.2f, 0.85f, 0.01f, 2 });
	network.propogate(input).printValues("Trained: ");
}

void testMNIST()
{
	// Read training dataset
	size_t trainImageCount, trainImageSize, trainLabelCount;
	uchar** trainImageDataset = MNIST::readImages("MNIST/train-images.idx3-ubyte", trainImageCount, trainImageSize);
	uchar* trainLabelDataset = MNIST::readLabels("MNIST/train-labels.idx1-ubyte", trainLabelCount);

	// Parse training dataset into input / training and cleanup
	tbml::Matrix trainInput = tbml::Matrix(trainImageCount, trainImageSize);
	tbml::Matrix trainExpected = tbml::Matrix(trainImageCount, 10);
	for (size_t i = 0; i < trainImageCount; i++)
	{
		for (size_t o = 0; o < trainImageSize; o++) trainInput(i, o) = (float)trainImageDataset[i][o] / 255.0f;
		trainExpected(i, trainLabelDataset[i]) = 1;
		delete trainImageDataset[i];
	}
	delete trainImageDataset;
	delete trainLabelDataset;

	// Read test dataset
	size_t testImageCount, testImageSize, testLabelCount;
	uchar** testImageDataset = MNIST::readImages("MNIST/t10k-images.idx3-ubyte", testImageCount, testImageSize);
	uchar* testLabelDataset = MNIST::readLabels("MNIST/t10k-labels.idx1-ubyte", testLabelCount);

	// Parse testing dataset into input / testing and cleanup
	tbml::Matrix testInput = tbml::Matrix(testImageCount, testImageSize);
	tbml::Matrix testExpected = tbml::Matrix(testImageCount, 10);
	for (size_t i = 0; i < testImageCount; i++)
	{
		for (size_t o = 0; o < testImageSize; o++) testInput(i, o) = (float)testImageDataset[i][o] / 255.0f;
		testExpected(i, testLabelDataset[i]) = 1;
		delete testImageDataset[i];
	}
	delete testImageDataset;
	delete testLabelDataset;

	// Print data info
	std::cout << std::endl;
	std::cout << "Training Image Count: " << trainImageCount << std::endl;
	trainInput.printDims("Training Input Dims: ");
	trainExpected.printDims("Training Expected Dims: ");
	std::cout << std::endl;
	std::cout << "Test Image Count: " << testImageCount << std::endl;
	testInput.printDims("Test Input Dims: ");
	testExpected.printDims("Test Expected Dims: ");
	std::cout << std::endl;

	// Optimising TanH + TanH + SE, batchSize = 128
	// -----------
	// Release x86	~90ms
	// Release x86	~65ms	Cross improvements (reverted)
	// Release x86	~42ms	1D matrix + omp 4 threaded cross
	// -----------

	// Epochs = 10, accuracy = 75.98%
	// tbml::SupervisedNetwork network({ trainImageSize, 100, 10 }, { tbml::fn::TanH(), tbml::fn::TanH() }, tbml::fn::SquareError());
	// std::cout << "Trainable Parameters: " << network.getParameterCount() << std::endl << std::endl;
	// network.train(trainInput, trainExpected, { 10, 128, 0.15f, 0.8f, 0.01f, 3 });

	// Epochs = 15, accuracy = 81.93%,920ms
	tbml::SupervisedNetwork network({ trainImageSize, 100, 10 }, { tbml::fn::ReLU(), tbml::fn::SoftMax() }, tbml::fn::CrossEntropy());
	network.train(trainInput, trainExpected, { 15, 50, 0.02f, 0.8f, 0.01f, 2 });

	// Epochs = 10, accuracy = 90.68%
	// tbml::SupervisedNetwork network({ trainImageSize, 200, 10 }, { tbml::fn::ReLU(), tbml::fn::SoftMax() }, tbml::fn::CrossEntropy());
	// network.train(trainInput, trainExpected, { 10, 128, 0.02f, 0.8f, 0.01f, 2 });

	// Epochs = 10, accuracy = 93.28%
	// tbml::SupervisedNetwork network({ trainImageSize, 250, 80, 10 }, { tbml::fn::ReLU(), tbml::fn::ReLU(), tbml::fn::SoftMax() }, tbml::fn::CrossEntropy());
	// std::cout << "Trainable Parameters: " << network.getParameterCount() << std::endl << std::endl;
	// network.train(trainInput, trainExpected, { 10, 128, 0.02f, 0.8f, 0.01f, 2 });

	// Test network against test data
	tbml::Matrix output = network.propogate(testInput);
	float accuracy = tbml::fn::calculateAccuracy(output, testExpected);
	std::cout << "t10k Accuracy = " << (accuracy * 100) << "%" << std::endl;
}
