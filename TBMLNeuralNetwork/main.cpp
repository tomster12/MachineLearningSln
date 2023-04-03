
#include <vector>
#include <iostream>
#include <chrono>

#include "Matrix.h"
#include "UtilityFunctions.h"
#include "SupervisedNetwork.h"
#include "MNIST.h"


void testBasic();
void testTime();
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
	tbml::NeuralNetwork network(std::vector<size_t>({ 3, 3, 1 }));
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
	tbml::NeuralNetwork network(std::vector<size_t>({ 8, 8, 8, 1 }));
	tbml::Matrix input = tbml::Matrix({ { 1, 0, -1, 0.2f, 0.7f, -0.3f, -1, -1 } });
	size_t epoch = 3'000'000;

	// Number of epochs propogation timing
	//
	// Release x86	1'000'000   ~1100ms
	// Release x86	1'000'000   ~600ms		Change to vector subscript from push_back
	// Release x86	3'000'000   ~1850ms
	//
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (size_t i = 0; i < epoch; i++) network.propogate(input);
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
	// Create network and setup training data
	tbml::SupervisedNetwork network(std::vector<size_t>({ 2, 2, 1 }), tbml::tanh, tbml::tanhPd, tbml::calcErrSqDiff, tbml::calcErrSqDiffPd);
	float l = -1.0f;
	float h = 1.0f;
	tbml::Matrix input = tbml::Matrix({
		{ l, l },
		{ l, h },
		{ h, l },
		{ h, h } });
	tbml::Matrix expected = tbml::Matrix({
		{ l },
		{ h },
		{ h },
		{ l } });

	// Print values and train
	input.printValues("Input:");
	expected.printValues("Expected:");
	network.propogate(input).printValues("Initial: ");
	network.train(input, expected, { -1, -1, 0.2f, 0.85f, 0.01f, 2 });
	network.propogate(input).printValues("Trained: ");
}

void testMNIST()
{
	// Read dataset
	size_t imageCount, imageSize, labelCount;
	uchar** imageDataset = MNIST::readImages("MNIST/train-images.idx3-ubyte", imageCount, imageSize);
	uchar* labelDataset = MNIST::readLabels("MNIST/train-labels.idx1-ubyte", labelCount);

	// Parse dataset into input / training
	std::vector<std::vector<float>> inputData = std::vector<std::vector<float>>(imageCount);
	std::vector<std::vector<float>> expectedData = std::vector<std::vector<float>>(imageCount);
	for (size_t i = 0; i < imageCount; i++)
	{
		inputData[i] = std::vector<float>(imageSize);
		expectedData[i] = std::vector<float>(10);
		for (size_t o = 0; o < imageSize; o++) inputData[i][o] = (float)imageDataset[i][o] / 255.0f;
		int label = labelDataset[i];
		expectedData[i][label] = 1;
	}
	tbml::Matrix input = tbml::Matrix(inputData);
	tbml::Matrix expected = tbml::Matrix(expectedData);

	// Print data
	std::cout << std::endl;
	input.printDims("Input Dims: ");
	expected.printDims("Expected Dims: ");
	std::cout << std::endl;

	// Batch size to time timing
	//
	// Release x86	128	~90ms 
	// 
	// For this to work in reasonable time will need:
	//	- GPU Parallelism
	//	- Small optimizations in Matrix class
	//	- More complex architecture involving convolutions
	//
	tbml::SupervisedNetwork network(std::vector<size_t>({ imageSize, 100, 10 }), tbml::tanh, tbml::tanhPd);
	network.train(input, expected, { 10, 128, 0.15f, 0.8f, 0.01f, 2 });
}
