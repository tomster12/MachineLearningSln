
#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class NeuralTargetGS;
class NeuralTargetGI : public tbml::GeneticInstance<NeuralGD>
{
public:
	NeuralTargetGI(NeuralTargetGI::DataPtr&& geneticData) : GeneticInstance(std::move(geneticData)) {};
	NeuralTargetGI(const NeuralTargetGS* sim, sf::Vector2f startPos, float radius, float moveAcc, int maxIterations, NeuralTargetGI::DataPtr&& geneticData);
	void initVisual();

	bool step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

private:
	const NeuralTargetGS* sim = nullptr;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	float radius = 0;
	float moveAcc = 0;
	int maxIterations = 0;
	sf::Vector2f pos;
	int currentIteration = 0;
};

class NeuralTargetGS : public tbml::GenepoolSimulation<NeuralGD, NeuralTargetGI>
{
public:
	NeuralTargetGS() {};
	NeuralTargetGS(
		sf::Vector2f instanceStartPos, float instanceRadius,
		float instancemoveAcc, int instancemaxIterations, std::vector<size_t> dataLayerSizes,
		float targetRadius, sf::Vector2f targetRandomCentre, float targetRandomRadius);

	void render(sf::RenderWindow* window) override;

	void initGeneration() override;

	sf::Vector2f getTargetPos() const;
	float getTargetRadius() const;

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

	DataPtr createData() const override;
	InstPtr createInstance(DataPtr&& data) const override;

	sf::Vector2f getRandomTargetPos() const;
};
