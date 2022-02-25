
#include <iostream>
#include <vector>
#include <ctime>
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "Activators.h"


NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes_) : NeuralNetwork(layerSizes_, actSigmoid) {}

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes_, float (*activator_)(float)) {
	// Initialize variables
	layerCount = layerSizes_.size();
	layerSizes = layerSizes_;
	activator = activator_;
	weights = std::vector<Matrix>();
	bias = std::vector<Matrix>();

	for (size_t layer = 0; layer < layerCount - 1; layer++) {
		
		// Initialize weights
		Matrix layerWeights = Matrix(layerSizes[layer], layerSizes[layer + 1]);
		weights.push_back(layerWeights.map([](float v) {
			return -5.0f + 10.0f * float(rand()) / float(RAND_MAX);
		}));

		// Initialize bias
		srand(time(NULL));
		Matrix layerBias = Matrix(1, layerSizes[layer + 1]);
		bias.push_back(layerBias.map([](float v) {
			return -5.0f + 10.0f * float(rand()) / float(RAND_MAX);
		}));
	}
}


Matrix NeuralNetwork::propogate(Matrix current) {
	// Intialize previous neuron outputs with current
	neuronOutCache = std::vector<Matrix>();
	neuronOutCache.push_back(current);
	
	// Run current through weight layers
	for (size_t layer = 0; layer < weights.size(); layer++) {
		current.icross(weights[layer]);
		current.iadd(bias[layer]); // Adds (1 x n) bias to each row of the (m x n) neuron values
		current.imap(activator);
		neuronOutCache.push_back(current);
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
