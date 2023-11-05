
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

VectorListGD::VectorListGD(std::vector<sf::Vector2f>&& values)
{
	this->dataSize = values.size();
	this->values = std::move(values);
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

	return std::make_shared<VectorListGD>(std::move(newValues));
}

#pragma endregion

#pragma region - NeuralGD

NeuralGD::NeuralGD(std::vector<size_t> layerSizes, float (*activator)(float))
	: network(layerSizes, activator, false)
{
	this->network.randomize();
}

NeuralGD::NeuralGD(tbml::NeuralNetwork&& network)
	: network(std::move(network))
{}

tbml::Matrix NeuralGD::propogate(tbml::Matrix& input) const { return this->network.propogate(input); }

void NeuralGD::print() const { this->network.printLayers(); }

NeuralGD::DataPtr NeuralGD::crossover(const NeuralGD::DataPtr& otherData, float mutateChance) const
{
	// Crossover weights
	const std::vector<tbml::Matrix>& weights = this->network.getWeights();
	const std::vector<tbml::Matrix>& oWeights = otherData->network.getWeights();

	std::vector<tbml::Matrix> newWeights = std::vector<tbml::Matrix>(weights.size());
	for (size_t i = 0; i < weights.size(); i++)
	{
		newWeights[i] = weights[i].ewise(oWeights[i], [mutateChance](float a, float b)
		{
			if (tbml::getRandomFloat() < mutateChance) return -1.0f + 2.0f * tbml::getRandomFloat();
			else return tbml::getRandomFloat() < 0.5f ? a : b;
		});
	}

	// Crossover bias
	const std::vector<tbml::Matrix>& bias = network.getBias();
	const std::vector<tbml::Matrix>& oBias = otherData->network.getBias();

	std::vector<tbml::Matrix> newBias = std::vector<tbml::Matrix>(bias.size());
	for (size_t i = 0; i < bias.size(); i++)
	{
		newBias[i] = bias[i].ewise(oBias[i], [mutateChance](float a, float b)
		{
			if (tbml::getRandomFloat() < mutateChance) return -1.0f + 2.0f * tbml::getRandomFloat();
			else return tbml::getRandomFloat() < 0.5f ? a : b;
		});
	}

	// Create new network and return
	tbml::NeuralNetwork network(std::move(newWeights), std::move(newBias), this->network.getActivator());
	return std::make_shared<NeuralGD>(std::move(network));
};

#pragma endregion
