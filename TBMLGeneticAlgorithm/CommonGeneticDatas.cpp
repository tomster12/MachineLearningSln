
#include "stdafx.h"
#include "CommonGeneticDatas.h"
#include "Matrix.h"

#pragma region - VectorListGD

VectorListGD::VectorListGD(int dataSize)
{
	// Initialize variables
	this->dataSize = dataSize;
	this->values = std::vector<sf::Vector2f>(dataSize);
}

VectorListGD::VectorListGD(std::vector<sf::Vector2f> values)
{
	// Initialize variables
	this->dataSize = values.size();
	this->values = values;
}

std::vector<sf::Vector2f>& VectorListGD::getValues() { return values; };

sf::Vector2f VectorListGD::getValue(int index) { return this->values[index]; }

size_t VectorListGD::getSize() { return this->dataSize; }

void VectorListGD::randomize()
{
	// Randomize each data point in the list
	for (int i = 0; i < this->dataSize; i++)
	{
		this->values[i].x = tbml::getRandomFloat() * 2 - 1;
		this->values[i].y = tbml::getRandomFloat() * 2 - 1;
	}
};

void VectorListGD::mutate(float chance)
{
	// Randomly mutate each data point in the list
	for (int i = 0; i < this->dataSize; i++)
	{
		if (tbml::getRandomFloat() < chance)
		{
			this->values[i].x = tbml::getRandomFloat() * 2 - 1;
			this->values[i].y = tbml::getRandomFloat() * 2 - 1;
		}
	}
};

VectorListGD* VectorListGD::crossover(VectorListGD* otherData)
{
	// Constructa new vector data
	std::vector<sf::Vector2f>& thisValues = this->getValues();
	std::vector<sf::Vector2f>& otherValues = otherData->getValues();
	std::vector<sf::Vector2f> newValues;
	newValues.reserve(this->getSize());

	for (size_t i = 0; i < this->getSize(); i++)
	{
		if (i % 2 == 0) newValues.push_back(thisValues[i]);
		else newValues.push_back(otherValues[i]);
	}

	return new VectorListGD(newValues);
}

VectorListGD* VectorListGD::clone()
{
	// Create a new data, pass values by value
	return new VectorListGD(values);
}

#pragma endregion

#pragma region - NeuralGD

NeuralGD::NeuralGD(std::vector<size_t> layerSizes, float (*activator)(float))
	: network(layerSizes, activator, false)
{}

NeuralGD::NeuralGD(tbml::NeuralNetwork network)
	: network(network)
{}

tbml::Matrix NeuralGD::propogate(tbml::Matrix& input) { return this->network.propogate(input); }

void NeuralGD::print() { this->network.printLayers(); }

void NeuralGD::randomize() { this->network.randomize(); };

void NeuralGD::mutate(float chance)
{
	// Mutate all weights
	// - Cannot use imap because cannot have capture chance into a lambda
	std::vector<tbml::Matrix>& weights = this->network.getWeights();
	for (auto& layer : weights)
	{
		std::vector<std::vector<float>>& data = layer.getData();
		for (size_t row = 0; row < layer.getRows(); row++)
		{
			for (size_t col = 0; col < layer.getCols(); col++)
			{
				if (tbml::getRandomFloat() < chance) data[row][col] = -1.0f + 2.0f * tbml::getRandomFloat();
			}
		}
	}

	// Mutate all bias
	std::vector<tbml::Matrix>& bias = this->network.getBias();
	for (auto& layer : bias)
	{
		std::vector<std::vector<float>>& data = layer.getData();
		for (size_t row = 0; row < layer.getRows(); row++)
		{
			for (size_t col = 0; col < layer.getCols(); col++)
			{
				if (tbml::getRandomFloat() < chance) data[row][col] = -1.0f + 2.0f * tbml::getRandomFloat();
			}
		}
	}
};

NeuralGD* NeuralGD::crossover(NeuralGD* otherData)
{
	// Crossover weights
	std::vector<tbml::Matrix>& weights = this->network.getWeights();
	std::vector<tbml::Matrix>& oWeights = otherData->network.getWeights();
	std::vector<tbml::Matrix> newWeights = std::vector<tbml::Matrix>();
	for (size_t i = 0; i < weights.size(); i++)
	{
		newWeights.push_back(weights[i].ewise(oWeights[i], [](float a, float b)
		{
			return tbml::getRandomFloat() < 0.5f ? a : b;
		}));
	}

	// Crossover bias
	std::vector<tbml::Matrix>& bias = network.getBias();
	std::vector<tbml::Matrix>& oBias = otherData->network.getBias();
	std::vector<tbml::Matrix> newBias = std::vector<tbml::Matrix>();
	for (size_t i = 0; i < bias.size(); i++)
	{
		newBias.push_back(bias[i].ewise(oBias[i], [](float a, float b)
		{
			return tbml::getRandomFloat() < 0.5f ? a : b;
		}));
	}

	// Create new network and return
	return new NeuralGD(tbml::NeuralNetwork(newWeights, newBias, network.getActivator()));
};

NeuralGD* NeuralGD::clone()
{
	// Create a new data, pass values by value
	return new NeuralGD(network);
}

#pragma endregion
