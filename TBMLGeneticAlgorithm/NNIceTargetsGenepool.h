#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class NNIceTargetsGenepool;
class NNIceTargetsAgent : public tbml::ga::Agent<NNGenome>
{
public:
	NNIceTargetsAgent(NNIceTargetsAgent::GenomeCPtr&& genome) : Agent(std::move(genome)) {};
	NNIceTargetsAgent(
		sf::Vector2f startPos, float radius, float moveAcc, float moveDrag, int maxIterations,
		const NNIceTargetsGenepool* genepool, NNIceTargetsAgent::GenomeCPtr&& genome);
	void initVisual();

	bool step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

private:
	const NNIceTargetsGenepool* genepool = nullptr;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	float radius = 0;
	float moveAcc = 0;
	float moveDrag = 0;
	int maxIterations = 0;

	tbml::Tensor netInput;
	sf::Vector2f pos;
	sf::Vector2f vel;
	int currentIteration = 0;
	int currentTarget = 0;
	float anger = 0.0f;
};

class NNIceTargetsGenepool : public tbml::ga::Genepool<NNGenome, NNIceTargetsAgent>
{
public:
	NNIceTargetsGenepool() {};
	NNIceTargetsGenepool(
		sf::Vector2f instanceStartPos, float instanceRadius, float instanceMoveAcc, float instanceMoveDrag, int instancemaxIterations,
		std::vector<sf::Vector2f> targets, float targetRadius,
		std::function<GenomeCPtr(void)> createGenomeFn);

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
	std::vector<sf::Vector2f> targetPos;
	float targetRadius = 0.0f;
	std::function<GenomeCPtr(void)> createGenomeFn;

	GenomeCPtr createGenome() const override;
	AgentPtr createAgent(GenomeCPtr&& data) const override;
};
