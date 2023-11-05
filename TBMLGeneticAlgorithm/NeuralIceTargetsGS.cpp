
#include "stdafx.h"
#include "global.h"
#include "NeuralIceTargetsGS.h"
#include "CommonImpl.h"
#include "UtilityFunctions.h"
#include "Matrix.h"

#pragma region - NeuralIceTargetsGI

NeuralIceTargetsGI::NeuralIceTargetsGI(const NeuralIceTargetsGS* sim, sf::Vector2f startPos, float radius, float moveAcc, float moveDrag, int maxIterations, NeuralIceTargetsGI::DataPtr&& geneticData)
	: GeneticInstance(std::move(geneticData)), sim(sim), netInput(1, 4), pos(startPos), radius(radius), moveAcc(moveAcc), moveDrag(moveDrag), maxIterations(maxIterations), currentIteration(0), currentTarget(0), vel(), anger(0.0f)
{
	if (global::showVisuals) initVisual();
}

void NeuralIceTargetsGI::initVisual()
{
	this->shape.setRadius(this->radius);
	this->shape.setOrigin(this->radius, this->radius);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}

bool NeuralIceTargetsGI::step()
{
	if (this->instanceFinished) return true;

	// Calculate with brain
	const sf::Vector2f& targetPos1 = this->sim->getTarget(this->currentTarget);
	const sf::Vector2f& targetPos2 = this->sim->getTarget(this->currentTarget + 1);
	netInput.set(0, 0, targetPos1.x - this->pos.x);
	netInput.set(0, 1, targetPos1.y - this->pos.y);
	netInput.set(0, 2, targetPos2.x - this->pos.x);
	netInput.set(0, 3, targetPos2.y - this->pos.y);
	netInput.set(0, 4, this->vel.x);
	netInput.set(0, 5, this->vel.y);
	tbml::Matrix output = this->geneticData->propogate(netInput);

	// Update position, velocity, drag
	this->vel.x += output.get(0, 0) * this->moveAcc * (1.0f / 60.0f);
	this->vel.y += output.get(0, 1) * this->moveAcc * (1.0f / 60.0f);
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
		this->instanceFinished = true;
	}
	return this->instanceFinished;
};

void NeuralIceTargetsGI::render(sf::RenderWindow* window)
{
	this->shape.setPosition(this->pos.x, this->pos.y);

	float fitness = this->calculateFitness();
	int v = static_cast<int>(255.0f * (0.3f + 0.7f * (fitness / 30.0f)));
	this->shape.setOutlineColor(sf::Color(v, v, v));

	window->draw(this->shape);
};

float NeuralIceTargetsGI::calculateDist()
{
	// Calculate distance to target
	sf::Vector2f targetPos = this->sim->getTarget(this->currentTarget);
	float dx = targetPos.x - pos.x;
	float dy = targetPos.y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->sim->getTargetRadius();
	return fullDistSq - radii - this->radius;
}

float NeuralIceTargetsGI::calculateFitness()
{
	// Dont calculate once finished
	if (this->instanceFinished) return this->instanceFitness;

	// Calculate fitness (anger)
	/*float fitness = std::min(1000000.0f / anger, 15.0f);
	fitness -= this->currentTarget * 2.0f;
	fitness = fitness > 0.0f ? fitness : 0.0f;*/

	// Calculate fitness (speed)
	float fitness = this->currentTarget + 1.0f - 1.0f / calculateDist();

	// Update and return
	this->instanceFitness = fitness;
	return this->instanceFitness;
};

#pragma endregion

#pragma region - NeuralIceTargetsGS

NeuralIceTargetsGS::NeuralIceTargetsGS(
	sf::Vector2f instanceStartPos, float instanceRadius, float instanceMoveAcc, float instanceMoveDrag,
	int instancemaxIterations, std::vector<size_t> dataLayerSizes,
	std::vector<sf::Vector2f> targets, float targetRadius, float (*activator)(float))
	: instanceStartPos(instanceStartPos), instanceRadius(instanceRadius), instanceMoveAcc(instanceMoveAcc), instanceMoveDrag(instanceMoveDrag),
	instancemaxIterations(instancemaxIterations), dataLayerSizes(dataLayerSizes),
	targetPos(targets), targetRadius(targetRadius), activator(activator)
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

NeuralIceTargetsGS::DataPtr NeuralIceTargetsGS::createData() const
{
	return std::make_shared<NeuralGD>(this->dataLayerSizes, this->activator);
};

NeuralIceTargetsGS::InstPtr NeuralIceTargetsGS::createInstance(NeuralIceTargetsGS::DataPtr&& data) const
{
	return std::make_unique<NeuralIceTargetsGI>(this, this->instanceStartPos, this->instanceRadius, this->instanceMoveAcc, this->instanceMoveDrag, this->instancemaxIterations, std::move(data));
};

void NeuralIceTargetsGS::render(sf::RenderWindow* window)
{
	GenepoolSimulation::render(window);

	// Draw target
	for (auto& shape : this->targetShapes) window->draw(shape);
}

const sf::Vector2f& NeuralIceTargetsGS::getTarget(int index) const { return this->targetPos[index % this->targetPos.size()]; }

size_t NeuralIceTargetsGS::getTargetCount() const { return this->targetPos.size(); }

float NeuralIceTargetsGS::getTargetRadius() const { return this->targetRadius; }

#pragma endregion
