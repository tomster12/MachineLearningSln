
#pragma once

#include <vector>
#include "Matrix.h"


class NeuralNetwork {

protected:
	size_t layerCount;
	std::vector<int> layerSizes;
	std::vector<Matrix> weights;
	std::vector<Matrix> bias;
	float (*activator)(float);
	std::vector<Matrix> neuronOutCache;

public:
	NeuralNetwork(std::vector<int> layerSizes_);
	NeuralNetwork(std::vector<int> layerSizes_, float (*activator_)(float));

	Matrix propogate(Matrix input);
	void printLayers();

	std::vector<Matrix>& getWeights();
	std::vector<Matrix>& getBias();
};
