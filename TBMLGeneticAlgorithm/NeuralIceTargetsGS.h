
#pragma once

#include "GenepoolSimulation.h"
#include "CommonGeneticDatas.h"

class NeuralIceTargetsGS;
class NeuralIceTargetsGI : public tbml::GeneticInstance<NeuralGD>
{
public:
	NeuralIceTargetsGI(NeuralGD* geneticData) : tbml::GeneticInstance<NeuralGD>(geneticData) {};
	NeuralIceTargetsGI(NeuralIceTargetsGS* sim, sf::Vector2f startPos, float moveAcc, int maxIterations, NeuralGD* geneticData);
	void initVisual();

	bool step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

private:
	NeuralIceTargetsGS* sim = nullptr;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	float moveAcc = 0;
	int maxIterations = 0;

	tbml::Matrix netInput;
	sf::Vector2f pos;
	sf::Vector2f vel;
	int currentIteration = 0;
	int currentTarget = 0;
	float anger = 0.0f;
};


class NeuralIceTargetsGS : public tbml::GenepoolSimulation<NeuralGD, NeuralIceTargetsGI>
{
public:
	NeuralIceTargetsGS() {};
	NeuralIceTargetsGS(
		sf::Vector2f instanceStartPos, float instancemoveAcc,
		int instancemaxIterations, std::vector<size_t> dataLayerSizes,
		std::vector<sf::Vector2f> targets, float targetRadius);

	void render(sf::RenderWindow* window) override;

	sf::Vector2f getTarget(int index);
	size_t getTargetCount();
	float getTargetRadius();

protected:
	std::vector<sf::CircleShape> targetShapes;
	sf::Vector2f instanceStartPos;
	float instancemoveAcc = 0.0f;
	int instancemaxIterations = 0;
	std::vector<size_t> dataLayerSizes;
	std::vector<sf::Vector2f> targetPos;
	float targetRadius = 0.0f;

	NeuralGD* createData() override;
	NeuralIceTargetsGI* createInstance(NeuralGD* data) override;
};
