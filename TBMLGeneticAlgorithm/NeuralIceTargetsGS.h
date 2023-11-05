
#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class NeuralIceTargetsGS;
class NeuralIceTargetsGI : public tbml::GeneticInstance<NeuralGD>
{
public:
	NeuralIceTargetsGI(NeuralIceTargetsGI::DataPtr&& geneticData) : GeneticInstance(std::move(geneticData)) {};
	NeuralIceTargetsGI(const NeuralIceTargetsGS* sim, sf::Vector2f startPos, float radius, float moveAcc, float moveDrag, int maxIterations, NeuralIceTargetsGI::DataPtr&& geneticData);
	void initVisual();

	bool step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

private:
	const NeuralIceTargetsGS* sim = nullptr;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	float radius = 0;
	float moveAcc = 0;
	float moveDrag = 0;
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
		sf::Vector2f instanceStartPos, float instanceRadius, float instanceMoveAcc, float instanceMoveDrag,
		int instancemaxIterations, std::vector<size_t> dataLayerSizes,
		std::vector<sf::Vector2f> targets, float targetRadius, float (*activator)(float) = tbml::tanh);

	void render(sf::RenderWindow* window) override;

	const sf::Vector2f& getTarget(int index) const;
	size_t getTargetCount() const;
	float getTargetRadius() const;

protected:
	std::vector<sf::CircleShape> targetShapes;
	sf::Vector2f instanceStartPos;
	float instanceRadius = 0.0f;
	float instanceMoveAcc = 0.0f;
	float instanceMoveDrag = 0.0f;
	int instancemaxIterations = 0;
	std::vector<size_t> dataLayerSizes;
	std::vector<sf::Vector2f> targetPos;
	float targetRadius = 0.0f;
	float (*activator)(float);

	DataPtr createData() const override;
	InstPtr createInstance(DataPtr&& data) const override;
};
