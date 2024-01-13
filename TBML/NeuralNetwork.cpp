#include "stdafx.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include "Utility.h"

namespace tbml
{
	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes, WeightInitType weightInitType)
		: layerCount(layerSizes.size()), layerSizes(layerSizes), weights(), bias(), actFns(layerCount - 1)
	{
		// Use single default activator for all layers
		for (size_t i = 0; i < layerCount; i++) this->actFns[i] = fns::Sigmoid();

		InitializeWeights(weightInitType);
	}

	NeuralNetwork::NeuralNetwork(std::vector<size_t> layerSizes, std::vector<fns::ActivationFunction> actFns, WeightInitType weightInitType)
		: layerCount(layerSizes.size()), layerSizes(layerSizes), weights(), bias(), actFns(actFns)
	{
		InitializeWeights(weightInitType);
	}

	NeuralNetwork::NeuralNetwork(std::vector<Matrix> weights, std::vector<Matrix> bias, std::vector<fns::ActivationFunction> actFns)
		: layerCount(weights.size() + 1), layerSizes(), weights(weights), bias(bias), actFns(actFns)
	{
		// Use passed in weights to calculate layer sizes
		for (size_t i = 0; i < this->layerCount - 1; i++) this->layerSizes.push_back(this->weights[i].getRowCount());
		this->layerSizes.push_back(this->weights[this->layerCount - 2].getColCount());
	}

	void NeuralNetwork::InitializeWeights(WeightInitType type)
	{
		weights.reserve(layerCount - 1);
		bias.reserve(layerCount - 1);
		for (size_t layer = 0; layer < layerCount - 1; layer++)
		{
			weights.push_back(Matrix(layerSizes[layer], layerSizes[layer + 1]));
			bias.push_back(Matrix(1, layerSizes[layer + 1]));
		}

		if (type == RANDOM)
		{
			for (auto& layer : this->weights) layer.map([](float v) { return -1.0f + 2.0f * fns::getRandomFloat(); });
			for (auto& layer : this->bias) layer.map([](float v) { return -1.0f + 2.0f * fns::getRandomFloat(); });
		}
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
			current.addBounded(bias[layer]);
			actFns[layer](current);
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
