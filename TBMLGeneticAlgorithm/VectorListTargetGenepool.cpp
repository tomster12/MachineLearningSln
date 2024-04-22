#include "stdafx.h"
#include "VectorListTargetGenepool.h"
#include "CommonImpl.h"
#include "Utility.h"

#pragma region - VectorListTargetAgent

VectorListTargetAgent::VectorListTargetAgent(
	sf::Vector2f startPos, float radius, float moveAcc,
	const VectorListTargetGenepool* genepool, VectorListTargetAgent::GenomeCPtr&& genome)
	: Agent(std::move(genome)), genepool(genepool), pos(startPos), radius(radius), moveAcc(moveAcc), currentIndex(0)
{
	if (global::showVisuals) initVisual();
}

void VectorListTargetAgent::initVisual()
{
	this->shape.setRadius(this->radius);
	this->shape.setOrigin(this->radius, this->radius);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}

bool VectorListTargetAgent::step()
{
	if (this->isFinished) return true;

	// Move position by current vector
	sf::Vector2f nextDir = this->genome->getValue(this->currentIndex);
	this->pos.x += nextDir.x * this->moveAcc;
	this->pos.y += nextDir.y * this->moveAcc;
	this->currentIndex++;

	// Check finish conditions
	float dist = calculateDist();
	if (this->currentIndex == this->genome->getSize() || dist < 0.0f)
	{
		this->calculateFitness();
		this->isFinished = true;
	}
	return this->isFinished;
};

void VectorListTargetAgent::render(sf::RenderWindow* window)
{
	// Update shape position and colour
	this->shape.setPosition(this->pos.x, this->pos.y);

	// Draw shape to window
	window->draw(this->shape);
};

float VectorListTargetAgent::calculateDist()
{
	// Calculate distance to target
	float dx = this->genepool->getTargetPos().x - pos.x;
	float dy = this->genepool->getTargetPos().y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->radius + this->genepool->getTargetRadius();
	return fullDistSq - radii;
}

float VectorListTargetAgent::calculateFitness()
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
		float dataPct = static_cast<float>(this->currentIndex) / static_cast<float>(this->genome->getSize());
		fitness = 1.0f - 0.5f * dataPct;
	}

	// Update and return
	this->fitness = fitness;
	return this->fitness;
};

#pragma endregion

#pragma region - VectorListTargetGenepool

VectorListTargetGenepool::VectorListTargetGenepool(
	sf::Vector2f targetPos, float targetRadius,
	std::function<GenomeCPtr(void)> createGenomeFn,
	std::function<AgentPtr(GenomeCPtr&&)> createAgentFn)
	: createGenomeFn(createGenomeFn), createAgentFn(createAgentFn)
{
	// Initialize variables
	this->target.setRadius(this->targetRadius);
	this->target.setOrigin(this->targetRadius, this->targetRadius);
	this->target.setFillColor(sf::Color::Transparent);
	this->target.setOutlineColor(sf::Color::White);
	this->target.setOutlineThickness(1.0f);
	this->target.setPosition(this->targetPos);
};

VectorListTargetGenepool::GenomeCPtr VectorListTargetGenepool::createGenome() const
{
	return createGenomeFn();
};

VectorListTargetGenepool::AgentPtr VectorListTargetGenepool::createAgent(VectorListTargetGenepool::GenomeCPtr&& genome) const
{
	return createAgentFn(std::move(genome));
};

void VectorListTargetGenepool::render(sf::RenderWindow* window)
{
	Genepool::render(window);

	// Draw target
	window->draw(this->target);
}

sf::Vector2f VectorListTargetGenepool::getTargetPos() const { return this->targetPos; }

float VectorListTargetGenepool::getTargetRadius() const { return this->targetRadius; }

#pragma endregion
