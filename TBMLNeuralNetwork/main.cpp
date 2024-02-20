#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>

#include "MNIST.h"
#include "ThreadPool.h"
#include "Utility.h"
#include "Tensor.h"
#include "NeuralNetwork.h"

void testTime();
void testBackprop();
void testMNIST();

int main()
{
	testMNIST();
	return 0;
}

void testTime()
{
	// Create network and input
	tbml::nn::NeuralNetwork network(tbml::fn::SquareError(), {
		new tbml::nn::DenseLayer(8, 8, tbml::fn::Sigmoid()),
		new tbml::nn::DenseLayer(8, 8, tbml::fn::Sigmoid()),
		new tbml::nn::DenseLayer(8, 1, tbml::fn::Sigmoid()) });

	tbml::Tensor input = tbml::Tensor({ { 1, 0, -1, 0.2f, 0.7f, -0.3f, -1, -1 } });
	size_t epoch = 1'000'000;

	// Time network propogation
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

void testBackprop()
{
	// Create network and input
	const float L = -1.0f, H = 1.0f;
	tbml::Tensor input{ std::vector<std::vector<float>>{ { L, L }, { L, H }, { H, L }, { H, H } } };
	tbml::Tensor expected{ std::vector<std::vector<float>>{ { L }, { H }, { H }, { L } } };

	tbml::nn::NeuralNetwork network(tbml::fn::SquareError(), {
		new tbml::nn::DenseLayer(2, 2, tbml::fn::TanH()),
		new tbml::nn::DenseLayer(2, 1, tbml::fn::TanH()) });

	// Print values and train
	input.print("Input:");
	expected.print("Expected:");
	network.propogate(input).print("Net Initial: ");
	network.train(input, expected, { -1, -1, 0.2f, 0.85f, 0.01f, 2 });
	network.propogate(input).print("Net Trained: ");
}

void testMNIST()
{
	// Read training / test datasets
	size_t trainImageCount, trainImageSize, trainLabelCount;
	size_t testImageCount, testImageSize, testLabelCount;
	tbml::Tensor trainInput = MNIST::readImagesTensor("MNIST/train-images.idx3-ubyte", trainImageCount, trainImageSize);
	tbml::Tensor trainExpected = MNIST::readLabelsTensor("MNIST/train-labels.idx1-ubyte", trainLabelCount);
	tbml::Tensor testInput = MNIST::readImagesTensor("MNIST/t10k-images.idx3-ubyte", testImageCount, testImageSize);
	tbml::Tensor testExpected = MNIST::readLabelsTensor("MNIST/t10k-labels.idx1-ubyte", testLabelCount);
	assert(trainImageSize == 784 && testImageSize == 784);

	// Print out dataset information
	std::cout << "\nTraining Image Count: " << trainImageCount << std::endl;
	trainInput.print("\nTraining Input: ");
	trainExpected.print("\nTraining Expected: ");
	std::cout << "\nTest Image Count: " << testImageCount << std::endl;
	testInput.print("\nTest Input: ");
	testExpected.print("\nTest Expected: ");

	// Create network and train
	// Batch: ~13ms, Epoch: ~19200ms, Total: ~300s, Accuracy: 95.23%
	tbml::nn::NeuralNetwork network(tbml::fn::CrossEntropy(), {
		new tbml::nn::DenseLayer(784, 100, tbml::fn::ReLU()),
		new tbml::nn::DenseLayer(100, 10, tbml::fn::SoftMax()) });
	network.train(trainInput, trainExpected, { 15, 50, 0.02f, 0.8f, 0.01f, 3 });

	// Test network against test data
	tbml::Tensor testPredicted = network.propogate(testInput);
	float accuracy = tbml::fn::classificationAccuracy(testPredicted, testExpected);
	std::cout << "t10k Accuracy = " << (accuracy * 100) << "%" << std::endl;
}
