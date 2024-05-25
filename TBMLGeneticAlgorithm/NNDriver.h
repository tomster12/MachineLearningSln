#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class Body
{
public:
	sf::Vector2f pos;
	sf::Vector2f size;
	float rot = 0.0f;

	Body() = default;
	Body(sf::Vector2f pos, sf::Vector2f size, float rot = 0.0f) : pos(pos), size(size), rot(rot) {};

	void updateShape(sf::RectangleShape& shape) const;
	void recalculateVertices();
	bool isColliding(const Body& other) const;

private:
std::vector<sf::Vector2f> vertices; private:
	static bool overlapOnAxis(const std::vector<sf::Vector2f>& vertices1, const std::vector<sf::Vector2f>& vertices2, const sf::Vector2f& axis);
	static std::pair<float, float> projectVerticesOnAxis(const std::vector<sf::Vector2f>& vertices, const sf::Vector2f& axis);
};

class NNDriverGenepool;
class NNDriverAgent : public tbml::ga::Agent<NNGenome>
{
public:
	NNDriverAgent(NNDriverAgent::GenomeCPtr&& genome) : Agent(std::move(genome)) {};
	NNDriverAgent(
		NNDriverAgent::GenomeCPtr&& genome, const NNDriverGenepool* genepool,
		sf::Vector2f startPos, float maxDrivingSpeed, float drivingAcc, float steeringSpeed, float moveDrag, float eyeLength, int maxIterations);

	void initVisual();
	void setFinishedVisual();
	bool evaluate() override;
	void render(sf::RenderWindow* window) override;
	void calculateFitness();

private:

	const NNDriverGenepool* genepool = nullptr;
	bool isVisualInit = false;
	Body mainBody;
	std::vector<Body> eyeBodies;
	sf::RectangleShape mainShape;
	std::vector<sf::RectangleShape> eyeShapes;
	sf::Color eyeColourHit;
	sf::Color eyeColourMiss;

	float maxDrivingSpeed = 2.0f;
	float steeringSpeed = 0.1f;
	float drivingAcc = 0.1f;
	float moveDrag = 0.999f;
	float eyeLength = 100.0f;
	int maxIterations = 0;

	int currentIteration = 0;
	tbml::Tensor netInput;
	float eyeHits[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float drivingSpeed = 0.0f;
	bool isFinished = false;
	int finishType = -1;
};

class NNDriverGenepool : public tbml::ga::Genepool<NNGenome, NNDriverAgent>
{
public:
	NNDriverGenepool(
		std::function<GenomeCnPtr(void)> createGenomeFn, std::function<AgentPtr(GenomeCnPtr)> createAgentFn,
		sf::Vector2f targetPos, float targetRadius, std::vector<Body> worldBodies);

	void initVisual();
	void render(sf::RenderWindow* window) override;
	bool isColliding(Body& body) const;
	float getTargetDist(sf::Vector2f pos) const;
	float getTargetDir(sf::Vector2f pos) const;
	float getTargetRadius() const { return targetRadius; }

private:
	sf::Vector2f targetPos;
	float targetRadius = 0.0f;
	sf::CircleShape targetShape;
	std::vector<Body> worldBodies;
	std::vector<sf::RectangleShape> worldShapes;
};
