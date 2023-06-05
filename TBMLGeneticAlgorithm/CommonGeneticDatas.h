
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

	std::vector<sf::Vector2f>& getValues();
	sf::Vector2f getValue(int index);
	size_t getSize();

	void randomize() override;
	void mutate(float chance) override;
	VectorListGD* crossover(VectorListGD* otherData) override;
	VectorListGD* clone() override;

private:
	std::vector<sf::Vector2f> values;
	int dataSize = 0;
};

class NeuralGD : public tbml::GeneticData<NeuralGD>
{
public:
	NeuralGD() {};
	NeuralGD(std::vector<size_t> layerSizes, float (*activator)(float) = tbml::sigmoid);
	NeuralGD(tbml::NeuralNetwork network);

	tbml::Matrix& propogate(tbml::Matrix& input);
	float getCachedOutput(int num);
	void print();

	void randomize() override;
	void mutate(float chance) override;
	NeuralGD* crossover(NeuralGD* otherData) override;
	NeuralGD* clone() override;

private:
	tbml::NeuralNetwork network;
};
