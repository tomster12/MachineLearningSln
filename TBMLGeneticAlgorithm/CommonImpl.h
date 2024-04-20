#pragma once

#include "GenepoolSimulation.h"
#include "NeuralNetwork.h"
#include "Tensor.h"

class VectorListGenome : public tbml::ga::Genome<VectorListGenome>
{
public:
	VectorListGenome() {}
	VectorListGenome(int dataSize);
	VectorListGenome(std::vector<sf::Vector2f>&& values);

	VectorListGenome::GenomeCPtr crossover(const VectorListGenome::GenomeCPtr& otherGenome, float chance) const override;
	const std::vector<sf::Vector2f>& getValues() const;
	const sf::Vector2f getValue(int index) const;
	const size_t getSize() const;

private:
	std::vector<sf::Vector2f> values;
	int dataSize = 0;
};

class NNGenome : public tbml::ga::Genome<NNGenome>
{
public:
	NNGenome() {};
	NNGenome(tbml::fn::LossFunctionPtr&& lossFn);
	NNGenome(tbml::fn::LossFunctionPtr&& lossFn, std::vector<std::shared_ptr<tbml::nn::Layer>>&& layers);

	NNGenome::GenomeCPtr crossover(const NNGenome::GenomeCPtr& otherData, float mutateChance) const override;
	tbml::Tensor propogate(const tbml::Tensor& input) const;
	const tbml::nn::NeuralNetwork& getNetwork() const { return this->network; }
	size_t getInputSize() const { return this->network.getInputShape()[0]; }
	void print() const;

private:
	tbml::nn::NeuralNetwork network;
};
