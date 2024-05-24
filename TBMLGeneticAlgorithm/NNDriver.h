#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class NNDriverGenepool;
class NNDriverAgent : public tbml::ga::Agent<NNGenome>
{
public:
	NNDriverAgent(NNDriverAgent::GenomeCPtr&& genome) : Agent(std::move(genome)) {};
	NNDriverAgent(
		NNDriverAgent::GenomeCPtr&& genome, const NNDriverGenepool* genepool,
		sf::Vector2f startPos, float maxDrivingSpeed, float drivingAcc, float steeringAcc, float moveDrag, float eyeLength,
		int maxIterations);

	void initVisual();
	bool evaluate() override;
	void render(sf::RenderWindow* window) override;
	void calculateFitness();

private:
	const NNDriverGenepool* genepool = nullptr;
	sf::RectangleShape bodyShape;
	std::vector<sf::RectangleShape> eyeShapes;
	float maxDrivingSpeed = 2.0f;
	float steeringAcc = 0.1f;
	float drivingAcc = 0.1f;
	float moveDrag = 0.999f;
	float eyeLength;
	int maxIterations = 0;
	int currentIteration = 0;
	tbml::Tensor netInput;
	float eyeHits[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	sf::Vector2f pos;
	float drivingAngle = 3.14159265f * 1.5f;
	float drivingSpeed = 0.0f;
	bool isFinished = false;
	bool hasReachedTarget = false;
};

class NNDriverGenepool : public tbml::ga::Genepool<NNGenome, NNDriverAgent>
{
public:
	NNDriverGenepool(
		std::function<GenomeCnPtr(void)> createGenomeFn, std::function<AgentPtr(GenomeCnPtr)> createAgentFn,
		sf::Vector2f targetPos, float targetRadius, std::vector<sf::RectangleShape> worldShapes);

	void initVisual();
	void render(sf::RenderWindow* window) override;
	bool isColliding(sf::RectangleShape& shape) const;
	float getTargetDist(sf::Vector2f pos) const;
	float getTargetDir(sf::Vector2f pos) const;

	static bool isColliding(const sf::RectangleShape& shape1, const sf::RectangleShape& shape2);

private:
	sf::Vector2f targetPos;
	float targetRadius = 0.0f;
	sf::CircleShape targetShape;
	std::vector<sf::RectangleShape> worldShapes;
};
