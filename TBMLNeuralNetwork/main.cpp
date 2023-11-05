
#include <vector>
#include <iostream>
#include <chrono>

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
	testBackprop();
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
	size_t epoch = 10'000'000;

	// Number of epochs propogation timing
	// -----------
	// Release x86	1'000'000   ~1100ms
	// Release x86	1'000'000   ~600ms		Change to vector subscript from push_back
	// Release x86	3'000'000   ~1900ms
	// Release x86	3'000'000   ~1780ms		Update where icross initialises matrix
	// Release x86	3'000'000   ~1370ms		(Reverted) Make icross storage matrix static (cannot const or thread)
	// Release x86	10'000'000	~6000ms
	// -----------
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	tbml::PropogateCache cache;
	for (size_t i = 0; i < epoch; i++) network.propogate(input, cache);
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
	tbml::SupervisedNetwork network(std::vector<size_t>({ 2, 2, 1 }),
		tbml::tanh, tbml::tanhPd,
		tbml::calcErrSqDiff, tbml::calcErrSqDiffPd
	);
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
	// Read dataset
	size_t imageCount, imageSize, labelCount;
	uchar** imageDataset = MNIST::readImages("MNIST/train-images.idx3-ubyte", imageCount, imageSize);
	uchar* labelDataset = MNIST::readLabels("MNIST/train-labels.idx1-ubyte", labelCount);

	// Parse dataset into input / training and cleanup
	tbml::Matrix input = tbml::Matrix(imageCount, imageSize);
	tbml::Matrix expected = tbml::Matrix(imageCount, 10);
	for (size_t i = 0; i < imageCount; i++)
	{
		for (size_t o = 0; o < imageSize; o++) input.set(i, o, (float)imageDataset[i][o] / 255.0f);
		expected.set(i, labelDataset[i], 1);
		delete imageDataset[i];
	}
	delete imageDataset;
	delete labelDataset;

	// Print data
	std::cout << std::endl;
	input.printDims("Input Dims: ");
	expected.printDims("Expected Dims: ");
	std::cout << std::endl;

	// Batch size to time timing
	// -----------
	// Release x86	128	~90ms 
	// Release x86	128	~65ms	Slight icross improvements
	// -----------
	// For this to work in reasonable time will need:
	//	- GPU Parallelism
	//	- Small optimizations in Matrix class
	//	- More complex architecture involving convolutions
	tbml::SupervisedNetwork network({ imageSize, 100, 10 }, tbml::tanh, tbml::tanhPd);
	network.train(input, expected, { 10, 128, 0.15f, 0.8f, 0.01f, 2 });
}
