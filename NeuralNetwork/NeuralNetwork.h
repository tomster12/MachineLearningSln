
#pragma once

#include <vector>
#include "Matrix.h"


class NeuralNetwork {

protected:
	// Protected variables
	size_t layerCount;
	std::vector<int> layerSizes;
	std::vector<Matrix> weights;
	std::vector<Matrix> bias;
	float (*activator)(float);
	std::vector<Matrix> neuronOutCache;

public:
	// Public constructors
	NeuralNetwork(std::vector<int> layerSizes_);
	NeuralNetwork(std::vector<int> layerSizes_, float (*activator_)(float));

	// Public functions
	Matrix propogate(Matrix input);
	void printLayers();
};
