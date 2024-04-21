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
void testSerialization();
void testMNIST();
void testMNISTSerialization();

int main()
{
	srand(0);
	testMNIST();
}

void testTime()
{
	// Create networks
	tbml::nn::NeuralNetwork network(std::make_shared<tbml::fn::SquareError>(), {
		std::make_shared<tbml::nn::DenseLayer>(8, 8, std::make_shared<tbml::fn::Sigmoid>()),
		std::make_shared<tbml::nn::DenseLayer>(8, 8, std::make_shared<tbml::fn::Sigmoid>()),
		std::make_shared<tbml::nn::DenseLayer>(8, 8, std::make_shared<tbml::fn::Sigmoid>()) });

	// Setup and print input
	tbml::Tensor input = tbml::Tensor({ { 1, 0, -1, 0.2f, 0.7f, -0.3f, -1, -1 } });
	input.print("Input: ");

	// Time different propogation methods
	size_t epoch = 5'000'000;
	std::chrono::steady_clock::time_point t00 = std::chrono::steady_clock::now();
	for (size_t i = 0; i < epoch; i++) network.propogate(input);
	std::chrono::steady_clock::time_point t01 = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point t10 = std::chrono::steady_clock::now();
	for (size_t i = 0; i < epoch; i++) network.propogateMut(input);
	std::chrono::steady_clock::time_point t11 = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point t20 = std::chrono::steady_clock::now();
	for (size_t i = 0; i < epoch; i++) network.propogateRef(input);
	std::chrono::steady_clock::time_point t21 = std::chrono::steady_clock::now();

	// Print output
	std::cout << std::endl << "Epochs: " << epoch << std::endl;
	float t0 = std::chrono::duration_cast<std::chrono::milliseconds>(t01 - t00).count() / (float)epoch;
	float t1 = std::chrono::duration_cast<std::chrono::milliseconds>(t11 - t10).count() / (float)epoch;
	float t2 = std::chrono::duration_cast<std::chrono::milliseconds>(t21 - t20).count() / (float)epoch;
	std::cout << "Prop: " << t0 << "ms" << std::endl;
	std::cout << "Prop Mut: " << t1 << "ms" << std::endl;
	std::cout << "Prop Ref: " << t2 << "ms" << std::endl;
}

void testBackprop()
{
	// Create network and input
	const float L = -1.0f, H = 1.0f;
	tbml::Tensor input{ std::vector<std::vector<float>>{ { L, L }, { L, H }, { H, L }, { H, H } } };
	tbml::Tensor expected{ std::vector<std::vector<float>>{ { L }, { H }, { H }, { L } } };

	tbml::nn::NeuralNetwork network(std::make_shared<tbml::fn::SquareError>(), {
		std::make_shared<tbml::nn::DenseLayer>(2, 2, std::make_shared<tbml::fn::TanH>()),
		std::make_shared<tbml::nn::DenseLayer>(2, 1, std::make_shared<tbml::fn::TanH>()) });

	// Print values and train
	input.print("Input:");
	expected.print("Expected:");
	network.propogate(input).print("Net Initial: ");
	network.train(input, expected, { -1, -1, 0.2f, 0.85f, 0.01f, 2 });
	network.propogate(input).print("Net Trained: ");
}

void testSerialization()
{
	tbml::nn::NeuralNetwork network(std::make_shared<tbml::fn::SquareError>(), {
		std::make_shared<tbml::nn::DenseLayer>(2, 2, std::make_shared<tbml::fn::ReLU>()),
		std::make_shared<tbml::nn::DenseLayer>(2, 1, std::make_shared<tbml::fn::Sigmoid>()) });

	network.print();

	network.saveToFile("test.nn");

	tbml::nn::NeuralNetwork network2 = tbml::nn::NeuralNetwork::loadFromFile("test.nn");

	network2.print();
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
	std::cout << "Training Image Count: " << trainImageCount << std::endl;
	trainInput.print("Training Input: ");
	trainExpected.print("Training Expected: ");
	std::cout << "Test Image Count: " << testImageCount << std::endl;
	testInput.print("Test Input: ");
	testExpected.print("Test Expected: ");

	// Create network and train
	// Timing: (Batch: ~6.7ms, Epoch: ~10100ms) @ 12 matmul threads, Fitness: (15 epochs = 95.22%, 50 epochs = 96.34%)
	tbml::nn::NeuralNetwork network(std::make_shared<tbml::fn::CrossEntropy>(), {
		std::make_shared<tbml::nn::DenseLayer>(784, 100, std::make_shared<tbml::fn::ReLU>()),
		std::make_shared<tbml::nn::DenseLayer>(100, 10, std::make_shared<tbml::fn::SoftMax>()) });
	std::cout << "Parameters: " << network.getParameterCount() << std::endl;
	network.train(trainInput, trainExpected, { 50, 50, 0.02f, 0.8f, 0.01f, 3 });

	// Test network against test data
	tbml::Tensor testPredicted = network.propogate(testInput);
	float accuracy = tbml::fn::classificationAccuracy(testPredicted, testExpected);
	std::cout << "t10k Accuracy = " << (accuracy * 100) << "%" << std::endl;

	// Save network to file
	network.saveToFile("MNIST.nn");
}

void testMNISTSerialization()
{
	// Read training / test datasets
	size_t trainImageCount, trainImageSize, trainLabelCount;
	size_t testImageCount, testImageSize, testLabelCount;
	tbml::Tensor trainInput = MNIST::readImagesTensor("MNIST/train-images.idx3-ubyte", trainImageCount, trainImageSize);
	tbml::Tensor trainExpected = MNIST::readLabelsTensor("MNIST/train-labels.idx1-ubyte", trainLabelCount);
	tbml::Tensor testInput = MNIST::readImagesTensor("MNIST/t10k-images.idx3-ubyte", testImageCount, testImageSize);
	tbml::Tensor testExpected = MNIST::readLabelsTensor("MNIST/t10k-labels.idx1-ubyte", testLabelCount);
	assert(trainImageSize == 784 && testImageSize == 784);

	// Read network from file
	tbml::nn::NeuralNetwork network = tbml::nn::NeuralNetwork::loadFromFile("MNIST.nn");

	// Print network information
	network.print();
	std::cout << "Parameters: " << network.getParameterCount() << std::endl;

	// Test network against test data
	tbml::Tensor testPredicted = network.propogate(testInput);
	float accuracy = tbml::fn::classificationAccuracy(testPredicted, testExpected);
	std::cout << "t10k Accuracy = " << (accuracy * 100) << "%" << std::endl;
}
