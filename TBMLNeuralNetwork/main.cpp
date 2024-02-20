﻿#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>

#include "MNIST.h"
#include "ThreadPool.h"
#include "Matrix.h"
#include "Utility.h"
#include "SupervisedNetwork.h"
#include "_Utility.h"
#include "_NeuralNetwork.h"
#include "_Tensor.h"

void testBasic();
void testTime();
void testTimeThreaded();
void testBackprop();
void testMNIST();
void testTimeNew();
void testNew();
void testMNISTNew();

int main()
{
	testNew();
	return 0;
}

void testBasic()
{
	// Create network, inputs, and run
	tbml::nn::NeuralNetwork network = tbml::nn::NeuralNetwork({ { 3, 3, 1 } }, { tbml::fn::Sigmoid(), tbml::fn::Sigmoid() });
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
	tbml::nn::NeuralNetwork network({ 8, 8, 8, 1 });
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
	tbml::nn::PropogateCache cache;
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
	tbml::nn::NeuralNetwork network(std::vector<size_t>({ 8, 8, 8, 1 }));
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
	tbml::nn::SupervisedNetwork network({ 2, 2, 1 }, { tbml::fn::TanH(), tbml::fn::TanH() }, tbml::fn::SquareError());
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

	// Optimising batch duration (TanH + TanH + SE, batchSize = 128)
	// -----------
	// Release x86	~90ms
	// Release x86	~65ms	Cross improvements (reverted)
	// Release x86	~42ms	1D matrix + omp 4 threaded cross
	// -----------
	// Epochs = 10, accuracy = 75.98%
	// tbml::nn::SupervisedNetwork network({ trainImageSize, 100, 10 }, { tbml::fn::TanH(), tbml::fn::TanH() }, tbml::fn::SquareError());
	// std::cout << "Trainable Parameters: " << network.getParameterCount() << std::endl << std::endl;
	// network.train(trainInput, trainExpected, { 10, 128, 0.15f, 0.8f, 0.01f, 3 });

	// Epochs = 15, accuracy = 81.93% (15ms batch duration)
	tbml::nn::SupervisedNetwork network({ trainImageSize, 100, 10 }, { tbml::fn::ReLU(), tbml::fn::SoftMax() }, tbml::fn::CrossEntropy());
	network.train(trainInput, trainExpected, { 15, 50, 0.02f, 0.8f, 0.01f, 3 });

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

void testTimeNew()
{
	// Create network and inputs
	tbml::nn::_NeuralNetwork network(tbml::fn::_SquareError(), {
		new tbml::nn::_DenseLayer(8, 8, tbml::fn::_Sigmoid()),
		new tbml::nn::_DenseLayer(8, 8, tbml::fn::_Sigmoid()),
		new tbml::nn::_DenseLayer(8, 1, tbml::fn::_Sigmoid()) });
	tbml::_Tensor input = tbml::_Tensor({ { 1, 0, -1, 0.2f, 0.7f, -0.3f, -1, -1 } });
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
	for (size_t i = 0; i < epoch; i++) network.propogate(input);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

	// Print output
	network.print();
	input.print("Input: ");
	std::cout << std::endl << "Epochs: " << epoch << std::endl;
	std::cout << "Time taken: " << us.count() / 1000.0f << "ms" << std::endl;
}

void testNew()
{
	// Setup training data
	const float L = -1.0f, H = 1.0f;

	tbml::Matrix input1 = tbml::Matrix({ { L, L }, { L, H }, { H, L }, { H, H } });
	tbml::Matrix expected1 = tbml::Matrix({ { L }, { H }, { H }, { L } });

	tbml::_Tensor input2{ std::vector<std::vector<float>>{ { L, L }, { L, H }, { H, L }, { H, H } } };
	tbml::_Tensor expected2{ std::vector<std::vector<float>>{ { L }, { H }, { H }, { L } } };

	// Setup old / new network
	tbml::nn::SupervisedNetwork network1(
		{ 2, 2, 1 },
		{ tbml::fn::TanH(), tbml::fn::TanH() },
		tbml::fn::SquareError());

	tbml::nn::_NeuralNetwork network2(tbml::fn::_SquareError(), {
		new tbml::nn::_DenseLayer(2, 2, tbml::fn::_TanH()),
		new tbml::nn::_DenseLayer(2, 1, tbml::fn::_TanH()) });

	// Print values and train
	input1.printValues("Input 1:");
	expected1.printValues("Expected 1:");
	network1.propogate(input1).printValues("Net1 Initial: ");

	input2.print("Input 2:");
	expected2.print("Expected 2:");
	network2.propogate(input2).print("Net2 Initial: ");

	network1.train(input1, expected1, { -1, -1, 0.2f, 0.85f, 0.01f, 2 });
	network2.train(input2, expected2, { -1, -1, 0.2f, 0.85f, 0.01f, 2 });

	network1.propogate(input1).printValues("Net1 Trained: ");
	network2.propogate(input2).print("Net2 Trained: ");
}

void testMNISTNew()
{
	// Read training dataset
	size_t trainImageCount, trainImageSize, trainLabelCount;
	uchar** trainImageDataset = MNIST::readImages("MNIST/train-images.idx3-ubyte", trainImageCount, trainImageSize);
	uchar* trainLabelDataset = MNIST::readLabels("MNIST/train-labels.idx1-ubyte", trainLabelCount);

	// Parse training dataset into input / training and cleanup
	tbml::_Tensor trainInput = tbml::_Tensor({ trainImageCount, trainImageSize }, 0);
	tbml::_Tensor trainExpected = tbml::_Tensor({ trainImageCount, 10 }, 0);
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
	tbml::_Tensor testInput = tbml::_Tensor({ testImageCount, testImageSize }, 0);
	tbml::_Tensor testExpected = tbml::_Tensor({ testImageCount, 10 }, 0);
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
	trainInput.print("Training Input: ");
	trainExpected.print("Training Expected: ");
	std::cout << std::endl;
	std::cout << "Test Image Count: " << testImageCount << std::endl;
	testInput.print("Test Input: ");
	testExpected.print("Test Expected: ");
	std::cout << std::endl;

	// OLD: Epochs = 15, accuracy = 81.93%  (15ms batch duration)
	// NEW: Epochs = 15, accuracy = XX.XX%  (XXms batch duration)
	tbml::nn::_NeuralNetwork network(tbml::fn::_CrossEntropy(), {
		new tbml::nn::_DenseLayer(784, 100, tbml::fn::_ReLU()),
		new tbml::nn::_DenseLayer(100, 10, tbml::fn::_SoftMax()) });

	network.train(trainInput, trainExpected, { 15, 50, 0.02f, 0.8f, 0.01f, 3 });

	// Test network against test data
	tbml::_Tensor testPredicted = network.propogate(testInput);
	float accuracy = tbml::fn::_classificationAccuracy(testPredicted, testExpected);
	std::cout << "t10k Accuracy = " << (accuracy * 100) << "%" << std::endl;
}
