
#include <iostream>
#include <vector>
#include <ctime>
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "Functions.h"


NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes_) : NeuralNetwork(layerSizes_, actSigmoid) {}

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes_, float (*activator_)(float)) {
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
		srand(time(NULL));
		Matrix layerBias = Matrix(1, layerSizes[layer + 1]);
		bias.push_back(layerBias.map([](float v) {
			return -1.0f + 2.0f * float(rand()) / float(RAND_MAX);
		}));
	}

	// Initialize weights as preset
	//std::vector<float> layer00 = std::vector<float>({ -4.183f, -5.877f });
	//std::vector<float> layer01 = std::vector<float>({ -4.182f, -5.868f });
	//std::vector<float> layer0bd = std::vector<float>({ 6.190f, 2.273f });
	//std::vector<float> layer10 = std::vector<float>({ 8.198f });
	//std::vector<float> layer11 = std::vector<float>({ -8.475f });
	//std::vector<float> layer1bd = std::vector<float>({ -3.766f });
	//Matrix layer0 = Matrix(std::vector<std::vector<float>>({ layer00, layer01 }));
	//Matrix layer0b = Matrix(std::vector<std::vector<float>>({ layer0bd }));
	//Matrix layer1 = Matrix(std::vector<std::vector<float>>({ layer10, layer11 }));
	//Matrix layer1b = Matrix(std::vector<std::vector<float>>({ layer1bd }));
	//weights = std::vector<Matrix>({ layer0, layer1 });
	//bias = std::vector<Matrix>({ layer0b, layer1b, });
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
