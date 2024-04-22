#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class NNTargetGenepool;
class NNTargetAgent : public tbml::ga::Agent<NNGenome>
{
public:
	NNTargetAgent(NNTargetAgent::GenomeCPtr&& genome) : Agent(std::move(genome)) {};
	NNTargetAgent(
		NNTargetAgent::GenomeCPtr&& genome, const NNTargetGenepool* genepool,
		sf::Vector2f startPos, float radius, float moveAcc, float moveDrag, int maxIterations);

	void initVisual();
	bool step() override;
	void render(sf::RenderWindow* window) override;
	float calculateDist();
	float calculateFitness();

private:
	const NNTargetGenepool* genepool = nullptr;
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

class NNTargetGenepool : public tbml::ga::Genepool<NNGenome, NNTargetAgent>
{
public:
	NNTargetGenepool() {};
	NNTargetGenepool(std::vector<sf::Vector2f> targets, float targetRadius);

	void initVisual();
	void render(sf::RenderWindow* window) override;
	const sf::Vector2f& getTarget(int index) const;
	size_t getTargetCount() const;
	float getTargetRadius() const;

protected:
	std::vector<sf::CircleShape> targetShapes;
	std::vector<sf::Vector2f> targetPos;
	float targetRadius = 0.0f;
};
