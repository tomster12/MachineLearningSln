
#include "stdafx.h"
/*#include "global.h"
#include "NeuralTargetGS.h"
#include "UtilityFunctions.h"
#include "CommonImpl.h"
#include "Matrix.h"

#pragma region - NeuralTargetGI

NeuralTargetGI::NeuralTargetGI(NeuralTargetGS* sim, sf::Vector2f startPos, float radius, float moveAcc, int maxIterations, NeuralGD* geneticData)
	: GeneticInstance(geneticData), sim(sim), pos(startPos), radius(radius), moveAcc(moveAcc), maxIterations(maxIterations), currentIteration(0)
{
	// Initialize variables
	if (global::showVisuals) initVisual();
}

void NeuralTargetGI::initVisual()
{
	// Initialize all visual variables
	this->shape.setRadius(this->radius);
	this->shape.setOrigin(this->radius, this->radius);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}

bool NeuralTargetGI::step()
{
	if (this->instanceFinished) return true;

	// Move position by current vector
	sf::Vector2f targetPos = this->sim->getTargetPos();
	tbml::Matrix input = tbml::Matrix({ { this->pos.x - targetPos.x, this->pos.y - targetPos.y } });
	tbml::Matrix output = this->geneticData->propogate(input);
	this->pos.x += output.get(0, 0) * this->moveAcc;
	this->pos.y += output.get(0, 1) * this->moveAcc;
	this->currentIteration++;

	// Check finish conditions
	float dist = calculateDist();
	if (this->currentIteration == this->maxIterations || dist < 0.0f)
	{
		this->calculateFitness();
		this->instanceFinished = true;
	}
	return this->instanceFinished;
};

void NeuralTargetGI::render(sf::RenderWindow* window)
{
	// Update shape position and colour
	this->shape.setPosition(this->pos.x, this->pos.y);

	// Draw shape to window
	window->draw(this->shape);
};

float NeuralTargetGI::calculateDist()
{
	// Calculate distance to target
	float dx = this->sim->getTargetPos().x - pos.x;
	float dy = this->sim->getTargetPos().y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->radius + this->sim->getTargetRadius();
	return fullDistSq - radii;
}

float NeuralTargetGI::calculateFitness()
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
		float dataPct = static_cast<float>(this->currentIteration) / static_cast<float>(this->maxIterations);
		fitness = 1.0f - 0.5f * dataPct;
	}

	// Update and return
	this->instanceFitness = fitness;
	return this->instanceFitness;
};

#pragma endregion

#pragma region - NeuralTargetGS

NeuralTargetGS::NeuralTargetGS(
	sf::Vector2f instanceStartPos, float instanceRadius,
	float instancemoveAcc, int instancemaxIterations, std::vector<size_t> dataLayerSizes,
	float targetRadius, sf::Vector2f targetRandomCentre, float targetRandomRadius)

	: instanceStartPos(instanceStartPos), instanceRadius(instanceRadius),
	instancemoveAcc(instancemoveAcc), instancemaxIterations(instancemaxIterations), dataLayerSizes(dataLayerSizes),
	targetRadius(targetRadius), targetRandomCentre(targetRandomCentre), targetRandomRadius(targetRandomRadius)
{
	// Initialize variables
	this->targetPos = this->getRandomTargetPos();
	this->target.setRadius(this->targetRadius);
	this->target.setOrigin(this->targetRadius, this->targetRadius);
	this->target.setFillColor(sf::Color::Transparent);
	this->target.setOutlineColor(sf::Color::White);
	this->target.setOutlineThickness(1.0f);
	this->target.setPosition(this->targetPos);
};

NeuralGD* NeuralTargetGS::createData()
{
	// Create, randomize and return data
	NeuralGD* data = new NeuralGD(this->dataLayerSizes);
	data->randomize();
	return data;
};

NeuralTargetGI* NeuralTargetGS::createInstance(NeuralGD* data)
{
	// Create and return instance
	NeuralTargetGI* inst = new NeuralTargetGI(this, this->instanceStartPos, this->instanceRadius, this->instancemoveAcc, this->instancemaxIterations, data);
	return inst;
};

void NeuralTargetGS::render(sf::RenderWindow* window)
{
	GenepoolSimulation::render(window);

	// Draw target
	window->draw(this->target);
}

void NeuralTargetGS::initGeneration()
{
	// Randomize target location
	this->targetPos = this->getRandomTargetPos();
	this->target.setPosition(this->targetPos);
}

sf::Vector2f NeuralTargetGS::getRandomTargetPos()
{
	// Return a random position within random target area
	return {
		this->targetRandomCentre.x + (tbml::getRandomFloat() * 2 - 1) * this->targetRandomRadius,
		this->targetRandomCentre.y
	};
}

sf::Vector2f NeuralTargetGS::getTargetPos() { return this->targetPos; }

float NeuralTargetGS::getTargetRadius() { return this->targetRadius; }

#pragma endregion
*/