#include "stdafx.h"
#include "CommonImpl.h"
#include "Tensor.h"

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

VectorListGenome::GenomeCPtr VectorListGenome::crossover(const VectorListGenome::GenomeCPtr& otherData, float mutateChance) const
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

const std::vector<sf::Vector2f>& VectorListGenome::getValues() const { return values; };

const sf::Vector2f VectorListGenome::getValue(int index) const { return this->values[index]; }

const size_t VectorListGenome::getSize() const { return this->dataSize; }

NNGenome::NNGenome(tbml::fn::LossFunctionPtr&& lossFn)
	: network(std::move(lossFn))
{}

NNGenome::NNGenome(tbml::fn::LossFunctionPtr&& lossFn, std::vector<std::shared_ptr<tbml::nn::Layer>>&& layers)
	: network(std::move(lossFn), std::move(layers))
{}

NNGenome::GenomeCPtr NNGenome::crossover(const NNGenome::GenomeCPtr& otherData, float mutateChance) const
{
	const std::vector<std::shared_ptr<tbml::nn::Layer>>& layers = this->network.getLayers();
	const std::vector<std::shared_ptr<tbml::nn::Layer>>& oLayers = otherData->network.getLayers();

	tbml::fn::LossFunctionPtr lossFunction = this->network.getLossFunction();
	std::vector<std::shared_ptr<tbml::nn::Layer>> newLayers(layers.size());

	for (size_t i = 0; i < layers.size(); i++)
	{
		// Pull out dense layers
		const tbml::nn::Layer& layer = *layers[i];
		const tbml::nn::Layer& oLayer = *oLayers[i];
		const tbml::nn::Dense& dLayer = dynamic_cast<const tbml::nn::Dense&>(layer);
		const tbml::nn::Dense& oDLayer = dynamic_cast<const tbml::nn::Dense&>(oLayer);

		// Perform crossover
		const tbml::Tensor& weights = dLayer.getWeights();
		const tbml::Tensor& oWeights = oDLayer.getWeights();
		const tbml::Tensor& biases = dLayer.getBias();
		const tbml::Tensor& oBiases = oDLayer.getBias();

		tbml::Tensor newWeights = weights.ewised(oWeights, [&](float a, float b) -> float
		{
			if (tbml::fn::getRandomFloat() < mutateChance) return tbml::fn::getRandomFloat() * 2 - 1;
			if (tbml::fn::getRandomFloat() < 0.5f) return a;
			return b;
		});

		tbml::Tensor newBiases = biases.ewised(oBiases, [&](float a, float b) -> float
		{
			if (tbml::fn::getRandomFloat() < mutateChance) return tbml::fn::getRandomFloat() * 2 - 1;
			if (tbml::fn::getRandomFloat() < 0.5f) return a;
			return b;
		});

		tbml::fn::ActivationFunctionPtr activationFn = dLayer.getActivationFunction();

		newLayers[i] = std::make_shared<tbml::nn::Dense>(std::move(newWeights), std::move(newBiases), std::move(activationFn));
	}

	return std::make_shared<NNGenome>(std::move(lossFunction), std::move(newLayers));
}

void NNGenome::print() const { this->network.print(); }
