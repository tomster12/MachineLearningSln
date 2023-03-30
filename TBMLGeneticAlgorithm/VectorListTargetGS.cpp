
#include "stdafx.h"
#include "VectorListTargetGS.h"
#include "UtilityFunctions.h"


#pragma region - VectorListGD

VectorListGD::VectorListGD(int dataSize)
{
	// Initialize variables
	this->dataSize = dataSize;
	this->values = std::vector<sf::Vector2f>(dataSize);
}

VectorListGD::VectorListGD(std::vector<sf::Vector2f> values)
{
	// Initialize variables
	this->dataSize = values.size();
	this->values = values;
}


std::vector<sf::Vector2f>& VectorListGD::getValues() { return values; };

sf::Vector2f VectorListGD::getValue(int index) { return this->values[index]; }

size_t VectorListGD::getSize() { return this->dataSize; }


void VectorListGD::randomize()
{
	// Randomize each data point in the list
	for (int i = 0; i < this->dataSize; i++)
	{
		this->values[i].x = tbml::getRandomFloat() * 2 - 1;
		this->values[i].y = tbml::getRandomFloat() * 2 - 1;
	}
};

void VectorListGD::mutate(float chance)
{
	// Randomly mutate each data point in the list
	for (int i = 0; i < this->dataSize; i++)
	{
		if (tbml::getRandomFloat() < chance)
		{
			this->values[i].x = tbml::getRandomFloat() * 2 - 1;
			this->values[i].y = tbml::getRandomFloat() * 2 - 1;
		}
	}
};

VectorListGD* VectorListGD::crossover(VectorListGD* otherData)
{
	// Constructa new vector data
	std::vector<sf::Vector2f>& thisValues = this->getValues();
	std::vector<sf::Vector2f>& otherValues = otherData->getValues();
	std::vector<sf::Vector2f> newValues;
	newValues.reserve(this->getSize());

	for (size_t i = 0; i < this->getSize(); i++)
	{
		if (i % 2 == 0) newValues.push_back(thisValues[i]);
		else newValues.push_back(otherValues[i]);
	}

	return new VectorListGD(newValues);
};

#pragma endregion


#pragma region - VectorListTargetGI

VectorListTargetGI::VectorListTargetGI(VectorListTargetGS* sim, sf::Vector2f startPos, float radius, float moveSpeed, VectorListGD* geneticData)
	: GeneticInstance(geneticData), sim(sim), pos(startPos), radius(radius), moveSpeed(moveSpeed), currentIndex(0)
{
	// Initialize variables
	if (global::showVisuals) initVisual();
}

void VectorListTargetGI::initVisual()
{
	// Initialize all visual variables
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
	this->pos.x += nextDir.x * this->moveSpeed;
	this->pos.y += nextDir.y * this->moveSpeed;
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

	} else
	{
		float dataPct = static_cast<float>(this->currentIndex) / static_cast<float>(this->geneticData->getSize());
		fitness = 1.0f - 0.5f * dataPct;
	}

	// Update and return
	this->instanceFitness = fitness;
	return this->instanceFitness;
};


bool VectorListTargetGI::getInstanceFinished() { return this->instanceFinished; };

float VectorListTargetGI::getInstanceFitness() { return calculateFitness(); };

#pragma endregion


#pragma region - VectorListTargetGS

VectorListTargetGS::VectorListTargetGS(
	sf::Vector2f instanceStartPos, float instanceRadius,
	float instanceMoveSpeed, int dataSize,
	sf::Vector2f targetPos, float targetRadius)
	: instanceStartPos(instanceStartPos), instanceRadius(instanceRadius),
	instanceMoveSpeed(instanceMoveSpeed), dataSize(dataSize),
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


VectorListGD* VectorListTargetGS::createData()
{
	// Create, randomize and return data
	VectorListGD* data = new VectorListGD(this->dataSize);
	data->randomize();
	return data;
};

VectorListTargetGI* VectorListTargetGS::createInstance(VectorListGD* data)
{
	// Create and return instance
	VectorListTargetGI* inst = new VectorListTargetGI(this, this->instanceStartPos, this->instanceRadius, this->instanceMoveSpeed, data);
	return inst;
};


void VectorListTargetGS::render(sf::RenderWindow* window)
{
	GenepoolSimulation::render(window);

	// Draw target
	window->draw(this->target);
}


sf::Vector2f VectorListTargetGS::getTargetPos() { return this->targetPos; }

float VectorListTargetGS::getTargetRadius() { return this->targetRadius; }

#pragma endregion
