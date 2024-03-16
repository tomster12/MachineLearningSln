#include "stdafx.h"
#include "global.h"
#include "NNTargetGenepool.h"
#include "Utility.h"
#include "CommonImpl.h"
#include "Tensor.h"

#pragma region - NNTargetAgent

NNTargetAgent::NNTargetAgent(
	sf::Vector2f startPos, float radius, float moveAcc, int maxIterations,
	const NNTargetGenepool* genepool, NNTargetAgent::GenomePtr&& genome)
	: Agent(std::move(genome)), genepool(genepool), pos(startPos), radius(radius), moveAcc(moveAcc), maxIterations(maxIterations), currentIteration(0)
{
	if (global::showVisuals) initVisual();
}

void NNTargetAgent::initVisual()
{
	// Initialize all visual variables
	this->shape.setRadius(this->radius);
	this->shape.setOrigin(this->radius, this->radius);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}

bool NNTargetAgent::step()
{
	if (this->isFinished) return true;

	// Move position by current vector
	sf::Vector2f targetPos = this->genepool->getTargetPos();
	tbml::Tensor input = tbml::Tensor({ { this->pos.x - targetPos.x, this->pos.y - targetPos.y } });
	tbml::Tensor output = this->genome->propogate(input);
	this->pos.x += output(0, 0) * this->moveAcc;
	this->pos.y += output(0, 1) * this->moveAcc;
	this->currentIteration++;

	// Check finish conditions
	float dist = calculateDist();
	if (this->currentIteration == this->maxIterations || dist < 0.0f)
	{
		this->calculateFitness();
		this->isFinished = true;
	}
	return this->isFinished;
};

void NNTargetAgent::render(sf::RenderWindow* window)
{
	// Update shape position and colour
	this->shape.setPosition(this->pos.x, this->pos.y);

	// Draw shape to window
	window->draw(this->shape);
};

float NNTargetAgent::calculateDist()
{
	// Calculate distance to target
	float dx = this->genepool->getTargetPos().x - pos.x;
	float dy = this->genepool->getTargetPos().y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->radius + this->genepool->getTargetRadius();
	return fullDistSq - radii;
}

float NNTargetAgent::calculateFitness()
{
	// Dont calculate once finished
	if (this->isFinished) return this->fitness;

	// Calculate fitness
	float dist = calculateDist();
	float fitness = 0.0f;

	if (dist > 0.0f)
	{
		fitness = 0.5f * (1.0f - dist / 500.0f);
		fitness = fitness < 0.0f ? 0.0f : fitness;
	}
	else
	{
		float dataPct = static_cast<float>(this->currentIteration) / static_cast<float>(this->maxIterations);
		fitness = 1.0f - 0.5f * dataPct;
	}

	// Update and return
	this->fitness = fitness;
	return this->fitness;
};

#pragma endregion

#pragma region - NNTargetGenepool

NNTargetGenepool::NNTargetGenepool(
	sf::Vector2f instanceStartPos, float instanceRadius, float instancemoveAcc, int instancemaxIterations,
	float targetRadius, sf::Vector2f targetRandomCentre, float targetRandomRadius,
	tbml::fn::LossFunction lossFn, std::vector<std::shared_ptr<tbml::nn::Layer>> layers)
	: instanceStartPos(instanceStartPos), instanceRadius(instanceRadius), instancemoveAcc(instancemoveAcc), instancemaxIterations(instancemaxIterations),
	targetRadius(targetRadius), targetRandomCentre(targetRandomCentre), targetRandomRadius(targetRandomRadius),
	lossFn(lossFn), layers(layers)
{
	// Initialize variables
	this->targetPos = this->getRandomTargetPos();
	this->target.setRadius(this->targetRadius);
	this->target.setOrigin(this->targetRadius, this->targetRadius);
	this->target.setFillColor(sf::Color::Transparent);
	this->target.setOutlineColor(sf::Color::White);
	this->target.setOutlineThickness(1.0f);
	this->target.setPosition(this->targetPos);
}

NNTargetGenepool::GenomePtr NNTargetGenepool::createGenome() const
{
	tbml::fn::LossFunction lossFn = this->lossFn;
	std::vector<std::shared_ptr<tbml::nn::Layer>> layers(this->layers.size());
	for (size_t i = 0; i < this->layers.size(); i++) layers[i] = this->layers[i]->clone();
	return std::make_shared<NNGenome>(std::move(lossFn), std::move(layers));
}

NNTargetGenepool::AgentPtr NNTargetGenepool::createAgent(NNTargetGenepool::GenomePtr&& genome) const
{
	return std::make_unique<NNTargetAgent>(this->instanceStartPos, this->instanceRadius, this->instancemoveAcc, this->instancemaxIterations, this, std::move(genome));
}

void NNTargetGenepool::render(sf::RenderWindow* window)
{
	Genepool::render(window);

	// Draw target
	window->draw(this->target);
}

void NNTargetGenepool::initGeneration()
{
	// Randomize target location
	this->targetPos = this->getRandomTargetPos();
	this->target.setPosition(this->targetPos);
}

sf::Vector2f NNTargetGenepool::getTargetPos() const { return this->targetPos; }

float NNTargetGenepool::getTargetRadius() const { return this->targetRadius; }

sf::Vector2f NNTargetGenepool::getRandomTargetPos() const
{
	// Return a random position within random target area
	return {
		this->targetRandomCentre.x + (tbml::fn::getRandomFloat() * 2 - 1) * this->targetRandomRadius,
		this->targetRandomCentre.y
	};
}

#pragma endregion
