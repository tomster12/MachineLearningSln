
#include "stdafx.h"
#include "VectorListTargetGS.h"
#include "CommonImpl.h"
#include "UtilityFunctions.h"

#pragma region - VectorListTargetGI

VectorListTargetGI::VectorListTargetGI(
	VectorListTargetGI::DataPtr&& geneticData, const VectorListTargetGS* sim, sf::Vector2f startPos, float radius, float moveAcc)
	: GeneticInstance(std::move(geneticData)), sim(sim), pos(startPos), radius(radius), moveAcc(moveAcc), currentIndex(0)
{
	if (global::showVisuals) initVisual();
}

void VectorListTargetGI::initVisual()
{
	this->shape.setRadius(this->radius);
	this->shape.setOrigin(this->radius, this->radius);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}

bool VectorListTargetGI::step()
{
	if (this->instanceFinished) return true;

	// Move position by current vector
	sf::Vector2f nextDir = this->geneticData->getValue(this->currentIndex);
	this->pos.x += nextDir.x * this->moveAcc;
	this->pos.y += nextDir.y * this->moveAcc;
	this->currentIndex++;

	// Check finish conditions
	float dist = calculateDist();
	if (this->currentIndex == this->geneticData->getSize() || dist < 0.0f)
	{
		this->calculateFitness();
		this->instanceFinished = true;
	}
	return this->instanceFinished;
};

void VectorListTargetGI::render(sf::RenderWindow* window)
{
	// Update shape position and colour
	this->shape.setPosition(this->pos.x, this->pos.y);

	// Draw shape to window
	window->draw(this->shape);
};

float VectorListTargetGI::calculateDist()
{
	// Calculate distance to target
	float dx = this->sim->getTargetPos().x - pos.x;
	float dy = this->sim->getTargetPos().y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->radius + this->sim->getTargetRadius();
	return fullDistSq - radii;
}

float VectorListTargetGI::calculateFitness()
{
	// Dont calculate once finished
	if (this->instanceFinished) return this->instanceFitness;

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
		float dataPct = static_cast<float>(this->currentIndex) / static_cast<float>(this->geneticData->getSize());
		fitness = 1.0f - 0.5f * dataPct;
	}

	// Update and return
	this->instanceFitness = fitness;
	return this->instanceFitness;
};

#pragma endregion

#pragma region - VectorListTargetGS

VectorListTargetGS::VectorListTargetGS(
	sf::Vector2f instanceStartPos, float instanceRadius,
	float instancemoveAcc, int dataSize,
	sf::Vector2f targetPos, float targetRadius)
	: instanceStartPos(instanceStartPos), instanceRadius(instanceRadius),
	instancemoveAcc(instancemoveAcc), dataSize(dataSize),
	targetPos(targetPos), targetRadius(targetRadius)
{
	// Initialize variables
	this->target.setRadius(this->targetRadius);
	this->target.setOrigin(this->targetRadius, this->targetRadius);
	this->target.setFillColor(sf::Color::Transparent);
	this->target.setOutlineColor(sf::Color::White);
	this->target.setOutlineThickness(1.0f);
	this->target.setPosition(this->targetPos);
};

VectorListTargetGS::DataPtr VectorListTargetGS::createData() const
{
	return std::make_shared<VectorListGD>(this->dataSize);
};

VectorListTargetGS::InstPtr VectorListTargetGS::createInstance(VectorListTargetGS::DataPtr&& data) const
{
	return std::make_unique<VectorListTargetGI>(std::move(data), this, this->instanceStartPos, this->instanceRadius, this->instancemoveAcc);
};

void VectorListTargetGS::render(sf::RenderWindow* window)
{
	GenepoolSimulation::render(window);

	// Draw target
	window->draw(this->target);
}

sf::Vector2f VectorListTargetGS::getTargetPos() const { return this->targetPos; }

float VectorListTargetGS::getTargetRadius() const { return this->targetRadius; }

#pragma endregion
