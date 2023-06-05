
#include "stdafx.h"
#include "global.h"
#include "NeuralIceTargetsGS.h"
#include "CommonGeneticDatas.h"
#include "UtilityFunctions.h"
#include "Matrix.h"

#pragma region - NeuralIceTargetsGI

NeuralIceTargetsGI::NeuralIceTargetsGI(NeuralIceTargetsGS* sim, sf::Vector2f startPos, float moveAcc, int maxIterations, NeuralGD* geneticData)
	: GeneticInstance(geneticData), sim(sim),
	netInput(1, 4),
	pos(startPos),
	moveAcc(moveAcc),
	maxIterations(maxIterations),
	currentIteration(0),
	currentTarget(0),
	vel(),
	anger(0.0f)
{
	if (global::showVisuals) initVisual();
}

void NeuralIceTargetsGI::initVisual()
{
	// Initialize all visual variables
	this->shape.setRadius(5.0f);
	this->shape.setOrigin(5.0f, 5.0f);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}

bool NeuralIceTargetsGI::step()
{
	if (this->instanceFinished) return true;

	// Move position by current vector
	sf::Vector2f targetPos = this->sim->getTarget(this->currentTarget);
	netInput.set(0, 0, this->pos.x - targetPos.x);
	netInput.set(0, 1, this->pos.y - targetPos.y);
	netInput.set(0, 2, this->vel.x);
	netInput.set(0, 3, this->vel.y);
	tbml::Matrix output = this->geneticData->propogate(netInput);
	this->vel.x += output.get(0, 0) * this->moveAcc * (1.0f / 60.0f);
	this->vel.y += output.get(0, 1) * this->moveAcc * (1.0f / 60.0f);
	this->currentIteration++;

	// Update position with velocity and apply drag
	this->pos.x += this->vel.x * (1.0f / 60.0f);
	this->pos.y += this->vel.y * (1.0f / 60.0f);
	this->vel.x *= 0.95f;
	this->vel.y *= 0.95f;

	// Check finish conditions
	float dist = calculateDist();
	anger += dist;
	if (dist <= 0.0f) this->currentTarget++;
	if (currentIteration == maxIterations || this->currentTarget == this->sim->getTargetCount())
	{
		this->calculateFitness();
		this->instanceFinished = true;
	}
	return this->instanceFinished;
};

void NeuralIceTargetsGI::render(sf::RenderWindow* window)
{
	// Update shape position and colour
	this->shape.setPosition(this->pos.x, this->pos.y);

	// Color based on fitness
	float fitness = this->calculateFitness();
	int v = static_cast<int>(255.0f * (0.3f + 0.7f * (fitness / 15.0f)));
	this->shape.setOutlineColor(sf::Color(v, v, v));

	// Draw shape to window
	window->draw(this->shape);
};

float NeuralIceTargetsGI::calculateDist()
{
	// Calculate distance to target
	if (this->currentTarget == this->sim->getTargetCount()) return 0.0f;
	sf::Vector2f targetPos = this->sim->getTarget(this->currentTarget);
	float dx = targetPos.x - pos.x;
	float dy = targetPos.y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->sim->getTargetRadius();
	return fullDistSq - radii - 5.0f;
}

float NeuralIceTargetsGI::calculateFitness()
{
	// Dont calculate once finished
	if (this->instanceFinished) return this->instanceFitness;

	// Calculate fitness (anger)
	float fitness = std::min(1000000.0f / anger, 15.0f);
	fitness -= this->currentTarget * 2.0f;
	fitness = fitness > 0.0f ? fitness : 0.0f;

	// Calculate fitness (speed)
	//float fitness = this->currentTarget + 1.0f;
	//if (this->currentTarget != this->sim->getTargetCount()) fitness -= (1.0f / calculateDist());
	//else fitness += 5.0f * (1.0f - ((float)this->currentIteration / this->maxIterations));

	// Update and return
	this->instanceFitness = fitness;
	return this->instanceFitness;
};

#pragma endregion

#pragma region - NeuralIceTargetsGS

NeuralIceTargetsGS::NeuralIceTargetsGS(
	sf::Vector2f instanceStartPos, float instancemoveAcc,
	int instancemaxIterations, std::vector<size_t> dataLayerSizes,
	std::vector<sf::Vector2f> targets, float targetRadius)
	: instanceStartPos(instanceStartPos), instancemoveAcc(instancemoveAcc),
	instancemaxIterations(instancemaxIterations), dataLayerSizes(dataLayerSizes),
	targetPos(targets), targetRadius(targetRadius)
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

NeuralGD* NeuralIceTargetsGS::createData()
{
	// Create, randomize and return data
	NeuralGD* data = new NeuralGD(this->dataLayerSizes, tbml::tanh);
	data->randomize();
	return data;
};

NeuralIceTargetsGI* NeuralIceTargetsGS::createInstance(NeuralGD* data)
{
	// Create and return instance
	NeuralIceTargetsGI* inst = new NeuralIceTargetsGI(this, this->instanceStartPos, this->instancemoveAcc, this->instancemaxIterations, data);
	return inst;
};


void NeuralIceTargetsGS::render(sf::RenderWindow* window)
{
	GenepoolSimulation::render(window);

	// Draw target
	for (auto& shape : this->targetShapes) window->draw(shape);
}

sf::Vector2f NeuralIceTargetsGS::getTarget(int index) { return this->targetPos[index]; }

size_t NeuralIceTargetsGS::getTargetCount() { return this->targetPos.size(); }

float NeuralIceTargetsGS::getTargetRadius() { return this->targetRadius; }

#pragma endregion
