#include "stdafx.h"
#include "global.h"
#include "NNTargetGenepool.h"
#include "CommonImpl.h"
#include "Utility.h"
#include "Tensor.h"

NNTargetAgent::NNTargetAgent(
	NNTargetAgent::GenomeCPtr&& genome, const NNTargetGenepool* genepool,
	sf::Vector2f startPos, float radius, float moveAcc, float moveDrag, int maxIterations)
	: Agent(std::move(genome)), genepool(genepool),
	pos(startPos), radius(radius), moveAcc(moveAcc), moveDrag(moveDrag), maxIterations(maxIterations),
	currentIteration(0), currentTarget(0), vel(), anger(0.0f), netInput({ 1, 6 }, 0.0f)
{
	if (global::showVisuals) initVisual();
}

void NNTargetAgent::initVisual()
{
	shape.setRadius(radius);
	shape.setOrigin(radius, radius);
	shape.setFillColor(sf::Color::Transparent);
	shape.setOutlineColor(sf::Color::White);
	shape.setOutlineThickness(1.0f);
}

bool NNTargetAgent::evaluate()
{
	if (isFinished) return true;

	// Calculate with brain
	const sf::Vector2f& targetPos1 = genepool->getTarget(currentTarget);
	const sf::Vector2f& targetPos2 = genepool->getTarget(currentTarget + 1);
	netInput.setData({ 1, 4 }, {
		targetPos1.x - pos.x,
		targetPos1.y - pos.y,
		vel.x,
		vel.y });
	genome->getNetwork().propogateMut(netInput);

	// Update position, velocity, drag
	vel.x += netInput(0, 0) * moveAcc * (1.0f / 60.0f);
	vel.y += netInput(0, 1) * moveAcc * (1.0f / 60.0f);
	pos.x += vel.x * (1.0f / 60.0f);
	pos.y += vel.y * (1.0f / 60.0f);
	vel.x *= moveDrag;
	vel.y *= moveDrag;
	currentIteration++;

	// Check finish conditions
	float dist = calculateDist();
	anger += dist;
	if (dist <= 0.0f) currentTarget++;
	if (currentIteration == maxIterations)
	{
		isFinished = true;
		this->calculateFitness();
	}
	return isFinished;
};

void NNTargetAgent::render(sf::RenderWindow* window)
{
	shape.setPosition(pos.x, pos.y);

	this->calculateFitness();
	int v = static_cast<int>(255.0f * (0.3f + 0.7f * (fitness / 30.0f)));
	shape.setOutlineColor(sf::Color(v, v, v));

	window->draw(shape);
};

float NNTargetAgent::calculateDist()
{
	// Calculate distance to target
	sf::Vector2f targetPos = genepool->getTarget(currentTarget);
	float dx = targetPos.x - pos.x;
	float dy = targetPos.y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = genepool->getTargetRadius();
	return fullDistSq - radii - radius;
}

float NNTargetAgent::calculateFitness()
{
	// Calculate fitness (anger)
	/*fitness = std::min(1000000.0f / anger, 15.0f);
	fitness -= currentTarget * 2.0f;
	fitness = fitness > 0.0f ? fitness : 0.0f;*/

	// Calculate fitness (speed)
	fitness = currentTarget + 1.0f - 1.0f / calculateDist();

	return fitness;
};

NNTargetGenepool::NNTargetGenepool(std::vector<sf::Vector2f> targets, float targetRadius)
	: targetPos(targets), targetRadius(targetRadius)
{
	if (global::showVisuals) initVisual();
};

void NNTargetGenepool::initVisual()
{
	// Initialize variables
	targetShapes = std::vector<sf::CircleShape>();
	for (auto& target : targetPos)
	{
		sf::CircleShape shape = sf::CircleShape();
		shape.setRadius(targetRadius);
		shape.setOrigin(targetRadius, targetRadius);
		shape.setFillColor(sf::Color::Transparent);
		shape.setOutlineColor(sf::Color::White);
		shape.setOutlineThickness(1.0f);
		shape.setPosition(target);
		targetShapes.push_back(shape);
	}
};

void NNTargetGenepool::render(sf::RenderWindow* window)
{
	Genepool::render(window);
	for (auto& shape : targetShapes) window->draw(shape);
}

const sf::Vector2f& NNTargetGenepool::getTarget(int index) const { return targetPos[index % targetPos.size()]; }

size_t NNTargetGenepool::getTargetCount() const { return targetPos.size(); }

float NNTargetGenepool::getTargetRadius() const { return targetRadius; }
