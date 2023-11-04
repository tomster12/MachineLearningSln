
#include "stdafx.h"
#include "CommonImpl.h"
#include "Matrix.h"

#pragma region - VectorListGD

VectorListGD::VectorListGD(int dataSize)
{
	this->dataSize = dataSize;
	this->values = std::vector<sf::Vector2f>(dataSize);
	for (int i = 0; i < this->dataSize; i++)
	{
		this->values[i].x = tbml::getRandomFloat() * 2 - 1;
		this->values[i].y = tbml::getRandomFloat() * 2 - 1;
	}
}

VectorListGD::VectorListGD(std::vector<sf::Vector2f> values)
{
	this->dataSize = values.size();
	this->values = values;
}

const std::vector<sf::Vector2f>& VectorListGD::getValues() const { return values; };

const sf::Vector2f VectorListGD::getValue(int index) const { return this->values[index]; }

const size_t VectorListGD::getSize() const { return this->dataSize; }

VectorListGD::DataPtr VectorListGD::crossover(const VectorListGD::DataPtr& otherData, float mutateChance) const
{
	std::vector<sf::Vector2f> newValues(this->getSize());

	for (size_t i = 0; i < this->getSize(); i++)
	{
		if (tbml::getRandomFloat() < mutateChance)
		{
			newValues[i].x = tbml::getRandomFloat() * 2 - 1;
			newValues[i].y = tbml::getRandomFloat() * 2 - 1;
		}
		else
		{
			if (i % 2 == 0) newValues[i] = this->values[i];
			else newValues[i] = otherData->values[i];
		}
	}

	return std::move(std::make_shared<VectorListGD>(newValues));
}

#pragma endregion

/*
#pragma region - NeuralGD

NeuralGD::NeuralGD(std::vector<size_t> layerSizes, float (*activator)(float))
	: network(layerSizes, activator, false)
{}

NeuralGD::NeuralGD(tbml::NeuralNetwork network)
	: network(network)
{}

tbml::Matrix NeuralGD::propogate(tbml::Matrix& input) { return this->network.propogate(input); }

void NeuralGD::print() const { this->network.printLayers(); }

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

NeuralGD::DataPtr NeuralGD::crossover(NeuralGD::DataPtr otherData) const
{
	// Crossover weights
	const std::vector<tbml::Matrix>& weights = this->network.getWeights();
	const std::vector<tbml::Matrix>& oWeights = otherData->network.getWeights();
	std::vector<tbml::Matrix> newWeights = std::vector<tbml::Matrix>();
	for (size_t i = 0; i < weights.size(); i++)
	{
		newWeights.push_back(weights[i].ewise(oWeights[i], [](float a, float b)
		{
			return tbml::getRandomFloat() < 0.5f ? a : b;
		}));
	}

	// Crossover bias
	const std::vector<tbml::Matrix>& bias = network.getBias();
	const std::vector<tbml::Matrix>& oBias = otherData->network.getBias();
	std::vector<tbml::Matrix> newBias = std::vector<tbml::Matrix>();
	for (size_t i = 0; i < bias.size(); i++)
	{
		newBias.push_back(bias[i].ewise(oBias[i], [](float a, float b)
		{
			return tbml::getRandomFloat() < 0.5f ? a : b;
		}));
	}

	// Create new network and return
	tbml::NeuralNetwork network(std::move(newWeights), std::move(newBias), this->network.getActivator());
	return std::make_shared<NeuralGD>(network);
};

NeuralGD::DataPtr NeuralGD::clone() const
{
	// Create a new data, pass values by value
	return std::make_shared<NeuralGD>(network);
}

#pragma endregion
*/
