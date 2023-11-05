
#pragma once

#include "GenepoolSimulation.h"
#include "NeuralNetwork.h"
#include "Matrix.h"

class VectorListGD : public tbml::GeneticData<VectorListGD>
{
public:
	VectorListGD() {}
	VectorListGD(int dataSize);
	VectorListGD(std::vector<sf::Vector2f>&& values);

	const std::vector<sf::Vector2f>& getValues() const;
	const sf::Vector2f getValue(int index) const;
	const size_t getSize() const;

	VectorListGD::DataPtr crossover(const VectorListGD::DataPtr& otherData, float chance) const override;

private:
	std::vector<sf::Vector2f> values;
	int dataSize = 0;
};

class NeuralGD : public tbml::GeneticData<NeuralGD>
{
public:
	NeuralGD() {};
	NeuralGD(std::vector<size_t> layerSizes, float (*activator)(float) = tbml::tanh);
	NeuralGD(tbml::NeuralNetwork&& network);

	tbml::Matrix propogate(tbml::Matrix& input) const;
	void print() const;

	NeuralGD::DataPtr crossover(const NeuralGD::DataPtr& otherData, float mutateChance) const override;
	size_t getInputSize() const { return this->network.getInputSize(); }

private:
	tbml::NeuralNetwork network;
};
