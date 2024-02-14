#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class VectorListTargetGenepool;
class VectorListTargetAgent : public tbml::ga::Agent<VectorListGenome>
{
public:
	VectorListTargetAgent(VectorListTargetAgent::GenomePtr&& genome) : Agent(std::move(genome)) {};
	VectorListTargetAgent(sf::Vector2f startPos, float radius, float moveAcc, const VectorListTargetGenepool* genepool, VectorListTargetAgent::GenomePtr&& genome);
	void initVisual();

	bool step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

private:
	const VectorListTargetGenepool* genepool = nullptr;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	sf::Vector2f pos;
	float moveAcc = 0.0f;
	float radius = 0.0f;
	int currentIndex = -1;
};

class VectorListTargetGenepool : public tbml::ga::Genepool<VectorListGenome, VectorListTargetAgent>
{
public:
	VectorListTargetGenepool() {};
	VectorListTargetGenepool(
		sf::Vector2f instanceStartPos, float instanceRadius, float instancemoveAcc,
		sf::Vector2f targetPos, float targetRadius,
		int dataSize);

	void render(sf::RenderWindow* window) override;

	sf::Vector2f getTargetPos() const;
	float getTargetRadius() const;

protected:
	sf::CircleShape target;
	sf::Vector2f instanceStartPos;
	float instanceRadius = 0.0f;
	sf::Vector2f targetPos;
	float targetRadius = 0.0f;
	float instancemoveAcc = 0.0f;
	int dataSize = 0;

	GenomePtr createGenome() const override;
	AgentPtr createAgent(GenomePtr&& genome) const override;
};
