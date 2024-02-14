#include "stdafx.h"
#include "CommonImpl.h"
#include "Matrix.h"

#pragma region - VectorListGenome

VectorListGenome::VectorListGenome(int dataSize)
{
	this->dataSize = dataSize;
	this->values = std::vector<sf::Vector2f>(dataSize);
	for (int i = 0; i < this->dataSize; i++)
	{
		this->values[i].x = tbml::fn::getRandomFloat() * 2 - 1;
		this->values[i].y = tbml::fn::getRandomFloat() * 2 - 1;
	}
}

VectorListGenome::VectorListGenome(std::vector<sf::Vector2f>&& values)
{
	this->dataSize = values.size();
	this->values = std::move(values);
}

const std::vector<sf::Vector2f>& VectorListGenome::getValues() const { return values; };

const sf::Vector2f VectorListGenome::getValue(int index) const { return this->values[index]; }

const size_t VectorListGenome::getSize() const { return this->dataSize; }

VectorListGenome::GenomePtr VectorListGenome::crossover(const VectorListGenome::GenomePtr& otherData, float mutateChance) const
{
	std::vector<sf::Vector2f> newValues(this->getSize());

	for (size_t i = 0; i < this->getSize(); i++)
	{
		if (tbml::fn::getRandomFloat() < mutateChance)
		{
			newValues[i].x = tbml::fn::getRandomFloat() * 2 - 1;
			newValues[i].y = tbml::fn::getRandomFloat() * 2 - 1;
		}
		else
		{
			if (i % 2 == 0) newValues[i] = this->values[i];
			else newValues[i] = otherData->values[i];
		}
	}

	return std::make_shared<VectorListGenome>(std::move(newValues));
}

#pragma endregion

#pragma region - NNGenome

NNGenome::NNGenome(std::vector<size_t> layerSizes)
	: network(layerSizes, tbml::nn::NeuralNetwork::WeightInitType::RANDOM)
{}

NNGenome::NNGenome(std::vector<size_t> layerSizes, std::vector<tbml::fn::ActivationFunction> actFns)
	: network(layerSizes, actFns, tbml::nn::NeuralNetwork::WeightInitType::RANDOM)
{}

NNGenome::NNGenome(tbml::nn::NeuralNetwork&& network)
	: network(std::move(network))
{}

tbml::Matrix NNGenome::propogate(tbml::Matrix& input) const { return this->network.propogate(input); }

void NNGenome::print() const { this->network.printLayers(); }

NNGenome::GenomePtr NNGenome::crossover(const NNGenome::GenomePtr& otherData, float mutateChance) const
{
	// Crossover weights
	const std::vector<tbml::Matrix>& weights = this->network.getWeights();
	const std::vector<tbml::Matrix>& oWeights = otherData->network.getWeights();

	std::vector<tbml::Matrix> newWeights = std::vector<tbml::Matrix>(weights.size());
	for (size_t i = 0; i < weights.size(); i++)
	{
		newWeights[i] = weights[i].ewised(oWeights[i], [mutateChance](float a, float b)
		{
			if (tbml::fn::getRandomFloat() < mutateChance) return -1.0f + 2.0f * tbml::fn::getRandomFloat();
			else return tbml::fn::getRandomFloat() < 0.5f ? a : b;
		});
	}

	// Crossover bias
	const std::vector<tbml::Matrix>& bias = network.getBias();
	const std::vector<tbml::Matrix>& oBias = otherData->network.getBias();

	std::vector<tbml::Matrix> newBias = std::vector<tbml::Matrix>(bias.size());
	for (size_t i = 0; i < bias.size(); i++)
	{
		newBias[i] = bias[i].ewised(oBias[i], [mutateChance](float a, float b)
		{
			if (tbml::fn::getRandomFloat() < mutateChance) return -1.0f + 2.0f * tbml::fn::getRandomFloat();
			else return tbml::fn::getRandomFloat() < 0.5f ? a : b;
		});
	}

	// Create new network and return
	tbml::nn::NeuralNetwork network(std::move(newWeights), std::move(newBias), this->network.getActivationFns());
	return std::make_shared<NNGenome>(std::move(network));
};

#pragma endregion
