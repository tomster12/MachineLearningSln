
#include "stdafx.h"
#include "global.h"
#include "NeuralRocketGS.h"
#include "UtilityFunctions.h"
#include "NeuralTargetGS.h"
#include "Matrix.h"


#pragma region - NeuralRocketGI

NeuralRocketGI::NeuralRocketGI(NeuralRocketGS* sim, sf::Vector2f startPos, float moveSpeed, int maxIterations, NeuralGD* geneticData)
	: GeneticInstance(geneticData), sim(sim), pos(startPos), moveSpeed(moveSpeed), maxIterations(maxIterations), currentIteration(0), currentTarget(0), vel() {

	// Initialize variables
	if (global::showVisuals) initVisual();
}

void NeuralRocketGI::initVisual() {
	// Initialize all visual variables
	this->shape.setRadius(5.0f);
	this->shape.setOrigin(5.0f, 5.0f);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}


void NeuralRocketGI::step() {
	if (this->instanceFinished) return;

	// Move position by current vector
	sf::Vector2f targetPos = this->sim->getTarget(this->currentTarget);
	tbml::Matrix input = tbml::Matrix({ {
		this->pos.x - targetPos.x,
		this->pos.y - targetPos.y,
		this->vel.x,
		this->vel.y
	} });
	tbml::Matrix output = this->geneticData->propogate(input);
	this->vel.x += output.get(0, 0) * this->moveSpeed * (1.0f / 120.0f);
	this->vel.y += output.get(0, 1) * this->moveSpeed * (1.0f / 120.0f);
	this->currentIteration++;

	// Update position with velocity and apply gravity
	this->vel.y += 0.02f;
	this->pos.x += this->vel.x;
	this->pos.y += this->vel.y;

	// Check finish conditions
	float dist = calculateDist();
	if (dist <= 0.0f) this->currentTarget++;
	if (currentIteration == maxIterations || this->currentTarget == this->sim->getTargetCount()) {
		this->calculateFitness();
		this->instanceFinished = true;
	}
};

void NeuralRocketGI::render(sf::RenderWindow* window) {
	// Update shape position and colour
	this->shape.setPosition(this->pos.x, this->pos.y);

	// Draw shape to window
	window->draw(this->shape);
};


float NeuralRocketGI::calculateDist() {
	// Calculate distance to target
	if (this->currentTarget == this->sim->getTargetCount()) return 0.0f;
	sf::Vector2f targetPos = this->sim->getTarget(this->currentTarget);
	float dx = targetPos.x - pos.x;
	float dy = targetPos.y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->sim->getTargetRadius();
	return fullDistSq - radii;
}

float NeuralRocketGI::calculateFitness() {
	// Dont calculate once finished
	if (this->instanceFinished) return this->instanceFitness;

	// Calculate fitness
	float dist = calculateDist();
	float fitness = 0.5f * tbml::relu(1.0f - dist / 500.0f);
	fitness += this->currentTarget;

	// Update and return
	this->instanceFitness = fitness;
	return this->instanceFitness;
};


bool NeuralRocketGI::getInstanceFinished() { return this->instanceFinished; };

float NeuralRocketGI::getInstanceFitness() { return calculateFitness(); };

#pragma endregion


#pragma region - NeuralRocketGS

NeuralRocketGS::NeuralRocketGS(
	sf::Vector2f instanceStartPos, float instanceMoveSpeed,
	int instancemaxIterations, std::vector<size_t> dataLayerSizes,
	std::vector<sf::Vector2f> targets, float targetRadius)

	: instanceStartPos(instanceStartPos), instanceMoveSpeed(instanceMoveSpeed),
	instancemaxIterations(instancemaxIterations), dataLayerSizes(dataLayerSizes),
	targetPos(targets), targetRadius(targetRadius) {

	// Initialize variables
	this->targetShapes = std::vector<sf::CircleShape>();
	for (auto& target : this->targetPos) {
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


NeuralGD* NeuralRocketGS::createData() {
	// Create, randomize and return data
	NeuralGD* data = new NeuralGD(this->dataLayerSizes);
	data->randomize();
	return data;
};

NeuralRocketGI* NeuralRocketGS::createInstance(NeuralGD* data) {
	// Create and return instance
	NeuralRocketGI* inst = new NeuralRocketGI(this, this->instanceStartPos, this->instanceMoveSpeed, this->instancemaxIterations, data);
	return inst;
};


void NeuralRocketGS::render(sf::RenderWindow* window) {
	GenepoolSimulation::render(window);

	// Draw target
	for (auto& shape : this->targetShapes) window->draw(shape);
}


sf::Vector2f NeuralRocketGS::getTarget(int index) { return this->targetPos[index]; }

float NeuralRocketGS::getTargetCount() { return this->targetPos.size(); }

float NeuralRocketGS::getTargetRadius() { return this->targetRadius; }

#pragma endregion
