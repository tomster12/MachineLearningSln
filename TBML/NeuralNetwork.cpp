
#include "stdafx.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "UtilityFunctions.h"


namespace tbml {

	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes_) : NeuralNetwork(layerSizes_, sigmoid) {}

	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float)) {
		// Initialize variables
		layerCount = layerSizes_.size();
		layerSizes = layerSizes_;
		activator = activator_;
		weights = std::vector<Matrix>();
		bias = std::vector<Matrix>();

		// Initialize weights as random within a range
		for (size_t layer = 0; layer < layerCount - 1; layer++) {

			// Initialize weights
			Matrix layerWeights = Matrix(layerSizes[layer], layerSizes[layer + 1]);
			weights.push_back(layerWeights.map([](float v) {
				return -1.0f + 2.0f * float(rand()) / float(RAND_MAX);
				}));

			// Initialize bias
			srand(static_cast<unsigned int>(time(NULL)));
			Matrix layerBias = Matrix(1, layerSizes[layer + 1]);
			bias.push_back(layerBias.map([](float v) {
				return -1.0f + 2.0f * float(rand()) / float(RAND_MAX);
				}));
		}
	}


	Matrix NeuralNetwork::propogate(Matrix current) {
		// Intialize previous neuron outputs with current
		if (neuronOutCache.size() != layerCount) neuronOutCache = std::vector<Matrix>(layerCount);
		neuronOutCache[0] = current;

		// Run current through weight layers
		for (size_t layer = 0; layer < weights.size(); layer++) {
			current.icross(weights[layer]);
			current.iadd(bias[layer]); // Adds (1 x n) bias to each row of the (m x n) neuron values
			current.imap(activator);
			neuronOutCache[layer + 1] = current;
		}

		// Return final layer values
		return current;
	}

	void NeuralNetwork::printLayers() {
		// Print layer / bias values
		std::cout << "\nLayers\n------" << std::endl;
		for (size_t layer = 0; layer < layerCount - 1; layer++)
			weights[layer].printValues(std::to_string(layer) + ": ");
		std::cout << "\nBias\n------" << std::endl;
		for (size_t layer = 0; layer < layerCount - 1; layer++)
			bias[layer].printValues(std::to_string(layer) + ": ");
		std::cout << "------\n" << std::endl;
	}


	std::vector<Matrix>& NeuralNetwork::getWeights() { return this->weights; }

	std::vector<Matrix>& NeuralNetwork::getBias() { return this->bias; }
}
