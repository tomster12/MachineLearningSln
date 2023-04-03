
#pragma once

#include "GenepoolSimulation.h"
#include "CommonGeneticDatas.h"


class NeuralTargetGS;
class NeuralTargetGI : public tbml::GeneticInstance<NeuralGD>
{
private:
	NeuralTargetGS* sim;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	float radius = 0;
	float moveAcc = 0;
	int maxIterations = 0;
	sf::Vector2f pos;
	int currentIteration = 0;


public:
	NeuralTargetGI(NeuralGD* geneticData) : tbml::GeneticInstance<NeuralGD>(geneticData) {};
	NeuralTargetGI(NeuralTargetGS* sim, sf::Vector2f startPos, float radius, float moveAcc, int maxIterations, NeuralGD* geneticData);
	void initVisual();

	bool step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();
};


class NeuralTargetGS : public tbml::GenepoolSimulation<NeuralGD, NeuralTargetGI>
{
protected:
	sf::CircleShape target;
	sf::Vector2f instanceStartPos;
	float instanceRadius = 0.0f;
	float instancemoveAcc = 0.0f;
	int instancemaxIterations = 0;
	std::vector<size_t> dataLayerSizes;
	float targetRadius = 0.0f;
	sf::Vector2f targetRandomCentre;
	float targetRandomRadius = 0.0f;
	sf::Vector2f targetPos;

	NeuralGD* createData() override;
	NeuralTargetGI* createInstance(NeuralGD* data) override;

	sf::Vector2f getRandomTargetPos();

public:
	NeuralTargetGS() {};
	NeuralTargetGS(
		sf::Vector2f instanceStartPos, float instanceRadius,
		float instancemoveAcc, int instancemaxIterations, std::vector<size_t> dataLayerSizes,
		float targetRadius, sf::Vector2f targetRandomCentre, float targetRandomRadius);

	void render(sf::RenderWindow* window) override;

	void initGeneration() override;

	sf::Vector2f getTargetPos();
	float getTargetRadius();
};
