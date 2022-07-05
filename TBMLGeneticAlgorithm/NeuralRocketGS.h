
#pragma once

#include "GenepoolSimulation.h"
#include "NeuralTargetGS.h"
#include "NeuralNetwork.h"
#include "Matrix.h"


class NeuralRocketGS;
class NeuralRocketGI : public tbml::GeneticInstance<NeuralGD> {

private:
	NeuralRocketGS* sim;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	float moveSpeed;
	int maxIterations;

	sf::Vector2f pos;
	sf::Vector2f vel;
	int currentIteration;
	int currentTarget;


public:
	NeuralRocketGI(NeuralGD* geneticData) : tbml::GeneticInstance<NeuralGD>(geneticData) {};
	NeuralRocketGI(NeuralRocketGS* sim, sf::Vector2f startPos, float moveSpeed, int maxIterations, NeuralGD* geneticData);
	void initVisual();

	void step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

	bool getInstanceFinished() override;
	float getInstanceFitness() override;
};


class NeuralRocketGS : public tbml::GenepoolSimulation<NeuralGD, NeuralRocketGI> {

protected:
	std::vector<sf::CircleShape> targetShapes;
	sf::Vector2f instanceStartPos;
	float instanceMoveSpeed;
	int instancemaxIterations;
	std::vector<size_t> dataLayerSizes;
	std::vector<sf::Vector2f> targetPos;
	float targetRadius;

	NeuralGD* createData() override;
	NeuralRocketGI* createInstance(NeuralGD* data) override;

public:
	NeuralRocketGS() {};
	NeuralRocketGS(
		sf::Vector2f instanceStartPos, float instanceMoveSpeed,
		int instancemaxIterations, std::vector<size_t> dataLayerSizes,
		std::vector<sf::Vector2f> targets, float targetRadius);

	void render(sf::RenderWindow* window) override;

	sf::Vector2f getTarget(int index);
	float getTargetCount();
	float getTargetRadius();
};
