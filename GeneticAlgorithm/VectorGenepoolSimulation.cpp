
#include "stdafx.h"
#include "Utility.h"
#include "VectorGenepoolSimulation.h"


#pragma region - Vector2fData

Vector2fData::Vector2fData(int dataSize) {
	// Initialize variables
	this->dataSize = dataSize;
	this->values = std::vector<sf::Vector2f>(dataSize);
}

Vector2fData::Vector2fData(std::vector<sf::Vector2f> values) {
	// Initialize variables
	this->dataSize = values.size();
	this->values = values;
}


std::vector<sf::Vector2f>& Vector2fData::getValues() { return values; };

sf::Vector2f Vector2fData::getValue(int index) { return this->values[index]; }

size_t Vector2fData::getSize() { return this->dataSize; }


void Vector2fData::randomize() {
	// Randomize each data point in the list
	for (int i = 0; i < this->dataSize; i++) {
		this->values[i].x = getRandomFloat() * 2 - 1;
		this->values[i].y = getRandomFloat() * 2 - 1;
	}
};

void Vector2fData::mutate(float chance) {
	// Randomly mutate each data point in the list
	for (int i = 0; i < this->dataSize; i++) {
		if (getRandomFloat() < chance) {
			this->values[i].x = getRandomFloat() * 2 - 1;
			this->values[i].y = getRandomFloat() * 2 - 1;
		}
	}
};

Vector2fData* Vector2fData::crossover(Vector2fData* otherData) {
	// Constructa new vector data
	std::vector<sf::Vector2f>& thisValues = this->getValues();
	std::vector<sf::Vector2f>& otherValues = otherData->getValues();
	std::vector<sf::Vector2f> newValues;
	newValues.reserve(this->getSize());

	for (size_t i = 0; i < this->getSize(); i++) {
		if (i % 2 == 0) newValues.push_back(thisValues[i]);
		else newValues.push_back(otherValues[i]);
	}

	return new Vector2fData(newValues);
};

#pragma endregion


#pragma region - VectorInstance

VectorInstance::VectorInstance(VectorGenepoolSimulation* sim, sf::Vector2f startPos, float radius, float moveSpeed, Vector2fData* data)
	: GeneticInstance(data), sim(sim), startPos(startPos), radius(radius), moveSpeed(moveSpeed) {

	// Initialize variables
	if (Globals::SHOW_VISUALS) initVisual();
	resetInstance();
}

void VectorInstance::initVisual() {
	// Initialize all visual variables
	this->shape.setRadius(this->radius);
	this->shape.setOrigin(this->radius, this->radius);
	this->shape.setFillColor(sf::Color::Transparent);
	this->shape.setOutlineColor(sf::Color::White);
	this->shape.setOutlineThickness(1.0f);
}

void VectorInstance::resetInstance() {
	// Reset variables
	this->pos = startPos;
	this->currentIndex = 0;
};


void VectorInstance::step() {
	if (this->instanceFinished) return;

	// Move position by current vector
	sf::Vector2f nextDir = this->geneticData->getValue(this->currentIndex);
	this->pos.x += nextDir.x * this->moveSpeed;
	this->pos.y += nextDir.y * this->moveSpeed;
	this->currentIndex++;

	// Check finish conditions
	float dist = calculateDist();
	if (this->currentIndex == this->geneticData->getSize() || dist < 0.0f) {
		this->calculateFitness();
		this->instanceFinished = true;
	}
};

void VectorInstance::render(sf::RenderWindow* window) {
	// Update shape position and colour
	this->shape.setPosition(this->pos.x, this->pos.y);

	// Draw shape to window
	window->draw(this->shape);
};


float VectorInstance::calculateDist() {
	// Calculate distance to target
	float dx = this->sim->getTargetPos().x - pos.x;
	float dy = this->sim->getTargetPos().y - pos.y;
	float fullDistSq = sqrt(dx * dx + dy * dy);
	float radii = this->radius + this->sim->getTargetRadius();
	return fullDistSq - radii;
}

float VectorInstance::calculateFitness() {
	// Dont calculate once finished
	if (this->instanceFinished) return this->instanceFitness;

	// Calculate fitness
	float dist = calculateDist();
	float fitness = 0.0f;

	if (dist > 0.0f) {
		fitness = 0.5f * (1.0f - dist / 500.0f);
		fitness = fitness < 0.0f ? 0.0f : fitness;

	} else {
		float dataPct = static_cast<float>(this->currentIndex) / static_cast<float>(this->geneticData->getSize());
		fitness = 1.0f - 0.5f * dataPct;
	}

	// Update and return
	this->instanceFitness = fitness;
	return this->instanceFitness;
};


bool VectorInstance::getInstanceFinished() { return this->instanceFinished; };

float VectorInstance::getInstanceFitness() { return calculateFitness(); };

#pragma endregion


#pragma region - VectorGenepoolSimulation

VectorGenepoolSimulation::VectorGenepoolSimulation(sf::Vector2f instanceStartPos, float instanceRadius, sf::Vector2f targetPos, float targetRadius, float instanceMoveSpeed, int dataSize)
	: instanceStartPos(instanceStartPos), instanceRadius(instanceRadius),
	targetPos(targetPos), targetRadius(targetRadius),
	instanceMoveSpeed(instanceMoveSpeed), dataSize(dataSize) {

	// Initialize variables
	this->target.setRadius(this->targetRadius);
	this->target.setOrigin(this->targetRadius, this->targetRadius);
	this->target.setFillColor(sf::Color::Transparent);
	this->target.setOutlineColor(sf::Color::White);
	this->target.setOutlineThickness(1.0f);
	this->target.setPosition(this->targetPos);
};


Vector2fData* VectorGenepoolSimulation::createData() {
	// Create, randomize and return data
	Vector2fData* data = new Vector2fData(this->dataSize);
	data->randomize();
	return data;
};

VectorInstance* VectorGenepoolSimulation::createInstance(Vector2fData* data) {
	// Create and return instance
	VectorInstance* inst = new VectorInstance(this, this->instanceStartPos, this->instanceRadius, this->instanceMoveSpeed, data);
	return inst;
};


void VectorGenepoolSimulation::render(sf::RenderWindow* window) {
	GenepoolSimulation::render(window);

	// Draw target
	window->draw(this->target);
}


sf::Vector2f VectorGenepoolSimulation::getTargetPos() { return this->targetPos; }

float VectorGenepoolSimulation::getTargetRadius() { return this->targetRadius; }

#pragma endregion
