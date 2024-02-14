#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class NNTargetGenepool;
class NNTargetAgent : public tbml::ga::Agent<NNGenome>
{
public:
	NNTargetAgent(NNTargetAgent::GenomePtr&& genome) : Agent(std::move(genome)) {};
	NNTargetAgent(
		sf::Vector2f startPos, float radius, float moveAcc, int maxIterations,
		const NNTargetGenepool* genepool, NNTargetAgent::GenomePtr&& genome);
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
	int maxIterations = 0;
	sf::Vector2f pos;
	int currentIteration = 0;
};

class NNTargetGenepool : public tbml::ga::Genepool<NNGenome, NNTargetAgent>
{
public:
	NNTargetGenepool() {};
	NNTargetGenepool(
		sf::Vector2f instanceStartPos, float instanceRadius, float instancemoveAcc, int instancemaxIterations,
		float targetRadius, sf::Vector2f targetRandomCentre, float targetRandomRadius,
		std::vector<size_t> layerSizes, std::vector<tbml::fn::ActivationFunction> actFns);

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
	float targetRadius = 0.0f;
	sf::Vector2f targetRandomCentre;
	float targetRandomRadius = 0.0f;
	sf::Vector2f targetPos;
	std::vector<size_t> layerSizes;
	std::vector<tbml::fn::ActivationFunction> actFns;

	GenomePtr createGenome() const override;
	AgentPtr createAgent(GenomePtr&& genome) const override;

	sf::Vector2f getRandomTargetPos() const;
};
