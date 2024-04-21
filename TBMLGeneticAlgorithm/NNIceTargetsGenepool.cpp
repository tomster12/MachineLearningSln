#include "stdafx.h"
#include "global.h"
#include "NNIceTargetsGenepool.h"
#include "CommonImpl.h"
#include "Utility.h"
#include "Tensor.h"

#pragma region - NNIceTargetsAgent

NNIceTargetsAgent::NNIceTargetsAgent(
	sf::Vector2f startPos, float radius, float moveAcc, float moveDrag, int maxIterations,
	const NNIceTargetsGenepool* genepool, NNIceTargetsAgent::GenomeCPtr&& genome)
	: pos(startPos), radius(radius), moveAcc(moveAcc), moveDrag(moveDrag), maxIterations(maxIterations),
	genepool(genepool), Agent(std::move(genome)),
	currentIteration(0), currentTarget(0), vel(), anger(0.0f),
	netInput({ 1, 6 }, 0.0f), network(this->genome->getNetwork())
{
	if (global::showVisuals) initVisual();
}

void NNIceTargetsAgent::initVisual()
{
	this->shape.setRadius(this->radius);
	this->shape.setOrigin(this->radius, this->radius);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}

bool NNIceTargetsAgent::step()
{
	if (this->isFinished) return true;

	// Calculate with brain
	const sf::Vector2f& targetPos1 = this->genepool->getTarget(this->currentTarget);
	const sf::Vector2f& targetPos2 = this->genepool->getTarget(this->currentTarget + 1);
	netInput(0, 0) = targetPos1.x - this->pos.x;
	netInput(0, 1) = targetPos1.y - this->pos.y;
	netInput(0, 2) = targetPos2.x - this->pos.x;
	netInput(0, 3) = targetPos2.y - this->pos.y;
	netInput(0, 4) = this->vel.x;
	netInput(0, 5) = this->vel.y;
	const tbml::Tensor& output = this->network.propogateMC(netInput);

	// Update position, velocity, drag
	this->vel.x += (output(0, 0) * 2 - 1) * this->moveAcc * (1.0f / 60.0f);
	this->vel.y += (output(0, 1) * 2 - 1) * this->moveAcc * (1.0f / 60.0f);
	this->pos.x += this->vel.x * (1.0f / 60.0f);
	this->pos.y += this->vel.y * (1.0f / 60.0f);
	this->vel.x *= this->moveDrag;
	this->vel.y *= this->moveDrag;
	this->currentIteration++;

	// Check finish conditions
	float dist = calculateDist();
	anger += dist;
	if (dist <= 0.0f) this->currentTarget++;
	if (currentIteration == maxIterations)
	{
		this->calculateFitness();
		this->isFinished = true;
	}
	return this->isFinished;
};

void NNIceTargetsAgent::render(sf::RenderWindow* window)
{
	this->shape.setPosition(this->pos.x, this->pos.y);

	float fitness = this->calculateFitness();
	int v = static_cast<int>(255.0f * (0.3f + 0.7f * (fitness / 30.0f)));
	this->shape.setOutlineColor(sf::Color(v, v, v));

	window->draw(this->shape);
};

float NNIceTargetsAgent::calculateDist()
{
	// Calculate distance to target
	sf::Vector2f targetPos = this->genepool->getTarget(this->currentTarget);
	float dx = targetPos.x - pos.x;
	float dy = targetPos.y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->genepool->getTargetRadius();
	return fullDistSq - radii - this->radius;
}

float NNIceTargetsAgent::calculateFitness()
{
	// Dont calculate once finished
	if (this->isFinished) return this->fitness;

	// Calculate fitness (anger)
	/*float fitness = std::min(1000000.0f / anger, 15.0f);
	fitness -= this->currentTarget * 2.0f;
	fitness = fitness > 0.0f ? fitness : 0.0f;*/

	// Calculate fitness (speed)
	float fitness = this->currentTarget + 1.0f - 1.0f / calculateDist();

	// Update and return
	this->fitness = fitness;
	return this->fitness;
};

#pragma endregion

#pragma region - NNIceTargetsGenepool

NNIceTargetsGenepool::NNIceTargetsGenepool(
	sf::Vector2f instanceStartPos, float instanceRadius, float instanceMoveAcc, float instanceMoveDrag, int instancemaxIterations,
	std::vector<sf::Vector2f> targets, float targetRadius,
	std::function<GenomeCPtr(void)> createGenomeFn)
	: instanceStartPos(instanceStartPos), instanceRadius(instanceRadius), instanceMoveAcc(instanceMoveAcc), instanceMoveDrag(instanceMoveDrag), instancemaxIterations(instancemaxIterations),
	targetPos(targets), targetRadius(targetRadius),
	createGenomeFn(createGenomeFn)
{
	// Initialize variables
	this->targetShapes = std::vector<sf::CircleShape>();
	for (auto& target : this->targetPos)
	{
		sf::CircleShape shape = sf::CircleShape();
		shape.setRadius(this->targetRadius);
		shape.setOrigin(this->targetRadius, this->targetRadius);
		shape.setFillColor(sf::Color::Transparent);
		shape.setOutlineColor(sf::Color::White);
		shape.setOutlineThickness(1.0f);
		shape.setPosition(target);
		this->targetShapes.push_back(shape);
	}
};

NNIceTargetsGenepool::GenomeCPtr NNIceTargetsGenepool::createGenome() const
{
	return createGenomeFn();
};

NNIceTargetsGenepool::AgentPtr NNIceTargetsGenepool::createAgent(NNIceTargetsGenepool::GenomeCPtr&& data) const
{
	return std::make_unique<NNIceTargetsAgent>(
		this->instanceStartPos, this->instanceRadius, this->instanceMoveAcc, this->instanceMoveDrag, this->instancemaxIterations,
		this, std::move(data));
};

void NNIceTargetsGenepool::render(sf::RenderWindow* window)
{
	Genepool::render(window);
	for (auto& shape : this->targetShapes) window->draw(shape);
}

const sf::Vector2f& NNIceTargetsGenepool::getTarget(int index) const { return this->targetPos[index % this->targetPos.size()]; }

size_t NNIceTargetsGenepool::getTargetCount() const { return this->targetPos.size(); }

float NNIceTargetsGenepool::getTargetRadius() const { return this->targetRadius; }

#pragma endregion
