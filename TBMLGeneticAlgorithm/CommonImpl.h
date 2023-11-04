
#pragma once

#include "GenepoolSimulation.h"
#include "NeuralNetwork.h"
#include "Matrix.h"

class VectorListGD : public tbml::GeneticData<VectorListGD>
{
public:
	VectorListGD() {}
	VectorListGD(int dataSize);
	VectorListGD(std::vector<sf::Vector2f> values);

	const std::vector<sf::Vector2f>& getValues() const;
	const sf::Vector2f getValue(int index) const;
	const size_t getSize() const;

	VectorListGD::DataPtr crossover(const VectorListGD::DataPtr& otherData, float chance) const override;

private:
	std::vector<sf::Vector2f> values;
	int dataSize = 0;
};

/*
class NeuralGD : public tbml::GeneticData<NeuralGD>
{
public:
	NeuralGD() {};
	NeuralGD(std::vector<size_t> layerSizes, float (*activator)(float) = tbml::sigmoid);
	NeuralGD(tbml::NeuralNetwork network);

	tbml::Matrix propogate(tbml::Matrix& input);
	void print() const;

	void randomize() override;
	void mutate(float chance) override;
	NeuralGD::DataPtr crossover(NeuralGD::DataPtr otherData) const override;
	NeuralGD::DataPtr clone() const override;

private:
	tbml::NeuralNetwork network;
};
*/
