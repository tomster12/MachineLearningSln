
#include "stdafx.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "UtilityFunctions.h"

namespace tbml
{
	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes, float (*activator)(float), bool toRandomize)
		: layerCount(layerSizes.size()), layerSizes(layerSizes), activator(activator), weights(), bias()
	{
		// Initialize w + b from preset sizes
		for (size_t layer = 0; layer < layerCount - 1; layer++)
		{
			weights.push_back(Matrix(layerSizes[layer], layerSizes[layer + 1]));
			bias.push_back(Matrix(1, layerSizes[layer + 1]));
		}

		// Randomize if needed
		if (toRandomize) this->randomize();
	}

	NeuralNetwork::NeuralNetwork(std::vector<Matrix> weights, std::vector<Matrix> bias, float (*activator)(float))
		: layerCount(weights.size() + 1), weights(weights), bias(bias), activator(activator)
	{
		// Initialize sizes from preset w + b
		this->layerSizes = std::vector<size_t>();
		for (size_t i = 0; i < this->layerCount - 1; i++) this->layerSizes.push_back(this->weights[i].getRows());
		this->layerSizes.push_back(this->weights[this->layerCount - 2].getCols());
	}

	void NeuralNetwork::randomize()
	{
		// Randomize all weights
		for (auto& layer : this->weights)
		{
			layer.imap([](float v)
			{
				return -1.0f + 2.0f * getRandomFloat();
			});
		}

		// Randomize all bias
		for (auto& layer : this->bias)
		{
			layer.imap([](float v)
			{
				return -1.0f + 2.0f * getRandomFloat();
			});
		}
	}

	Matrix NeuralNetwork::propogate(const Matrix& input) const
	{
		// Propogate and return last layer
		PropogateCache cache;
		propogate(input, cache);
		return cache.neuronOutput[layerCount - 1];
	}

	void NeuralNetwork::propogate(const Matrix& input, PropogateCache& cache) const
	{
		// Intialize previous neuron outputs with current
		if (cache.neuronOutput.size() != layerCount) cache.neuronOutput.resize(layerCount);
		Matrix current = input;
		cache.neuronOutput[0] = current;

		// Run current through weight layers
		for (size_t layer = 0; layer < weights.size(); layer++)
		{
			current.icross(weights[layer]);
			current.iadd(bias[layer]); // Adds (1 x n) bias to each row of the (m x n) neuron values
			current.imap(activator);
			cache.neuronOutput[layer + 1] = current;
		}
	}

	void NeuralNetwork::printLayers() const
	{
		// Print layer / bias values
		std::cout << "\nLayers\n------" << std::endl;
		for (size_t layer = 0; layer < layerCount - 1; layer++)
			weights[layer].printValues(std::to_string(layer) + ": ");
		std::cout << "\nBias\n------" << std::endl;
		for (size_t layer = 0; layer < layerCount - 1; layer++)
			bias[layer].printValues(std::to_string(layer) + ": ");
		std::cout << "------\n" << std::endl;
	}

	std::vector<Matrix>& NeuralNetwork::getWeights() { return weights; }

	std::vector<Matrix>& NeuralNetwork::getBias() { return bias; }

	afptr NeuralNetwork::getActivator() const { return activator; }

	size_t NeuralNetwork::getLayerCount() const { return layerCount; }

	std::vector<size_t> NeuralNetwork::getLayerSizes() const { return layerSizes; }
}
