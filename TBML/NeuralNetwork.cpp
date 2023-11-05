
#include "stdafx.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "Utility.h"

namespace tbml
{
	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes, float (*activator)(float), bool toRandomize)
		: layerCount(layerSizes.size()), layerSizes(layerSizes), weights(), bias(), activator(activator)
	{
		// Create w + b from layer sizes
		weights.reserve(layerCount - 1);
		bias.reserve(layerCount - 1);
		for (size_t layer = 0; layer < layerCount - 1; layer++)
		{
			weights.push_back(Matrix(layerSizes[layer], layerSizes[layer + 1]));
			bias.push_back(Matrix(1, layerSizes[layer + 1]));
		}

		if (toRandomize) this->randomize();
	}

	NeuralNetwork::NeuralNetwork(std::vector<Matrix> weights, std::vector<Matrix> bias, float (*activator)(float))
		: layerCount(weights.size() + 1), weights(weights), bias(bias), activator(activator)
	{
		// Create layer sizes from w + b
		this->layerSizes = std::vector<size_t>();
		for (size_t i = 0; i < this->layerCount - 1; i++) this->layerSizes.push_back(this->weights[i].getRowCount());
		this->layerSizes.push_back(this->weights[this->layerCount - 2].getColCount());
	}

	void NeuralNetwork::randomize()
	{
		for (auto& layer : this->weights) layer.map([](float v) { return -1.0f + 2.0f * getRandomFloat(); });
		for (auto& layer : this->bias) layer.map([](float v) { return -1.0f + 2.0f * getRandomFloat(); });
	}

	Matrix NeuralNetwork::propogate(const Matrix& input) const
	{
		PropogateCache cache;
		propogate(input, cache);
		return cache.neuronOutput[layerCount - 1];
	}

	void NeuralNetwork::propogate(const Matrix& input, PropogateCache& cache) const
	{
		Matrix current = input;
		cache.neuronOutput.resize(layerCount);
		cache.neuronOutput[0] = current;

		for (size_t layer = 0; layer < weights.size(); layer++)
		{
			current.cross(weights[layer]);
			current += bias[layer];
			current.map(activator);
			cache.neuronOutput[layer + 1] = current;
		}
	}

	void NeuralNetwork::printLayers() const
	{
		std::cout << "\nLayers\n------" << std::endl;
		for (size_t layer = 0; layer < layerCount - 1; layer++)
			weights[layer].printValues(std::to_string(layer) + ": ");
		std::cout << "\nBias\n------" << std::endl;
		for (size_t layer = 0; layer < layerCount - 1; layer++)
			bias[layer].printValues(std::to_string(layer) + ": ");
		std::cout << "------\n" << std::endl;
	}
}
