
#include <iostream>
#include <chrono>
#include "SupervisedNetwork.h"
#include "Functions.h"
#include "MNIST.h"


void testBasic();
void testTime();
void testBackprop();
void testMNIST();


int main() {
	testMNIST();
	return 0;
}


void testBasic() {
	// Create network, inputs, and run
	NeuralNetwork network(std::vector<int>( { 3, 3, 1 } ));
	Matrix input = Matrix( { { 1.0f, -1.0f, 1.0f } });
	Matrix output = network.propogate(input);

	// Print values
	network.printLayers();
	input.printValues("Input:");
	output.printValues("Output:");
}


void testTime() {
	// Create network and inputs
	NeuralNetwork network(std::vector<int>({ 8, 8, 8, 1 }));
	Matrix input = Matrix({ { 1, 0, -1, 0.2f, 0.7f, -0.3f, -1, -1 } });
	size_t epoch = 50'000;

	// Time n epoch of propogation
	// Pre			50,000		~3000ms
	// Vec			50,000		~2450ms
	// Vec+ref		50,000		~850ms
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < epoch; i++) network.propogate(input);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

	// Print output
	network.printLayers();
	input.printValues("Input: ");
	std::cout << std::endl << "Epochs: " << epoch << std::endl;
	std::cout << "Time taken: " << us.count() / 1000.0f << "ms" << std::endl;
}


void testBackprop() {
	// Create network and setup training data
	SupervisedNetwork network(std::vector<int>({ 2, 2, 1 }), actTanh, actTanhPd, calcErrSqDiff, calcErrSqDiffPd);
	float l = -1.0f;
	float h = 1.0f;
	Matrix input = Matrix({
		{ l, l },
		{ l, h },
		{ h, l },
		{ h, h } });
	Matrix expected = Matrix({
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


void testMNIST() {
	// Read dataset
	int imageCount, imageSize, labelCount;
	uchar** imageDataset = MNIST::readImages("MNIST/train-images.idx3-ubyte", imageCount, imageSize);
	uchar* labelDataset = MNIST::readLabels("MNIST/train-labels.idx1-ubyte", labelCount);

	// Parse dataset into input / training
	std::vector<std::vector<float>> inputData = std::vector<std::vector<float>>(imageCount);
	std::vector<std::vector<float>> expectedData = std::vector<std::vector<float>>(imageCount);
	for (size_t i = 0; i < imageCount; i++) {
		inputData[i] = std::vector<float>(imageSize);
		expectedData[i] = std::vector<float>(10);
		for (size_t o = 0; o < imageSize; o++) inputData[i][o] = (float)imageDataset[i][o] / 255.0f;
		int label = labelDataset[i];
		expectedData[i][label] = 1;
	}
	Matrix input = Matrix(inputData);
	Matrix expected = Matrix(expectedData);

	// Print data
	std::cout << std::endl;
	input.printDims("Input Dims: ");
	expected.printDims("Expected Dims: ");
	std::cout << std::endl;

	// Create network and train
	// Pre			128			~3000ms
	// Vec			128			~1300ms
	// 
	// For this to work in reasonable time will need:
	//	- GPU Parallelism
	//	- Small optimizations in Matrix class
	//	- More complex architecture involving convolutions
	SupervisedNetwork network(std::vector<int>({ imageSize, 100, 10 }), actTanh, actTanhPd);
	network.train(input, expected, { 10, 128, 0.15f, 0.8f, 0.01f, 2 });
}


// --------------------------------
//
// 	-- Forward Propogation --
// 
// - Overview -
// 
// Layer Sizes:  2, 4, 1
// Bias:		 True
// Data Count:	 3
// 
// 
// - Data Layout -
// 
// Input	= [ x00, x01 ]
//			  [ x10, x11 ]
//			  [ x20, x21 ]
// 
// 
// weights l1 = [ w00 w01 w02 w03 ]
//			    [ w10 w11 w12 w13 ]
// 
// bias l1	  = [ b0, b1, b2, b3 ]
// 
// Values l1  = [ x00, x01, x02, x03 ]
//				[ x10, x11, x12, x13 ]
//				[ x20, x21, x22, x23 ]
// 
// 
// weight l2  = [ w00 ]
//			    [ w10 ]
//			    [ w20 ]
//			    [ w30 ]
// 
// bias l2	  = [ b0 ]
// 
// Values l2  = [ x00 ]
//				[ x10 ]
//				[ x20 ]
// 
// --------------------------------


// --------------------------------
//
// 	-- Back Propogation --
// 
// (δE / δWᵢⱼ)		= (δE / δoⱼ) * (δoⱼ / δnetⱼ) * (δnetⱼ / δWᵢⱼ)
// 
// pdErrorToWeight	= pdErrorToIn  * pdInToWeight
// (δE / δWᵢⱼ)		= (δE / δnetⱼ) * (δnetⱼ / δWᵢⱼ)
//					= (δE / δnetⱼ) * Out[j]
// 
// pdErrorToIn		= pdErrorToOut * pdOutToIn			(Cache)
// (δE / δnetⱼ)		= (δE / δoⱼ)   * (δoⱼ / δnetᵢⱼ)
// 
// pdErrorToOut[last]	= predicted - expected
// (δE / δoⱼ)			= (δE / δy)
//						= (y - t)
// 
// pdErrorToOut[other]	= sum(weight[J] * pdErrorToIn[J])
// (δE / δoⱼ)			= Σ(δWᵢⱼ * (δE / δnetⱼ))
// 
// pdOutToIn
// (δoⱼ / δnetᵢⱼ)	= drvActivator(out[j])
// 
// Error
// E(y)				= 1/2 * Σ(t - y) ^ 2
//
// 
// -- Gradient Descent --
// 
// weight += derivative * learningRate
// weight += momentum * momentumRate
// momentum = derivative
// 
// --------------------------------
