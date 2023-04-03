
#include "stdafx.h"
#include "NeuralPoleBalancerGS.h"
#include "UtilityFunctions.h"


#pragma region - NeuralPoleBalancerGI

NeuralPoleBalancerGI::NeuralPoleBalancerGI(
	float cartMass, float poleMass, float poleLength, float force,
	float trackLimit, float angleLimit, float timeLimit,
	NeuralGD* geneticData)
	:	cartMass(cartMass), poleMass(poleMass), poleLength(poleLength), force(force),
		trackLimit(trackLimit), angleLimit(angleLimit), timeLimit(timeLimit),
		tbml::GeneticInstance<NeuralGD>(geneticData)
{
	this->initVisual();
	this->poleAngle = 0.1f;
}

void NeuralPoleBalancerGI::initVisual()
{
	// Initialize all visual variables
	this->cartShape.setSize({ 0.3f * METRE_TO_UNIT, 0.22f * METRE_TO_UNIT });
	this->cartShape.setOrigin(0.5f * (0.3f * METRE_TO_UNIT), 0.5f * (0.32f * METRE_TO_UNIT));
	this->cartShape.setFillColor(sf::Color::Transparent);
	this->cartShape.setOutlineColor(sf::Color::White);
	this->cartShape.setOutlineThickness(1.0f);

	this->poleShape.setSize({ 5.0f, this->poleLength * METRE_TO_UNIT * 2 });
	this->poleShape.setOrigin(0.5f * 5.0f, this->poleLength * METRE_TO_UNIT * 2);
	this->poleShape.setFillColor(sf::Color::Transparent);
	this->poleShape.setOutlineColor(sf::Color::White);
	this->poleShape.setOutlineThickness(1.0f);
}


bool NeuralPoleBalancerGI::step()
{
	if (this->instanceFinished) return true;

	// Calculate force with netowkr
	tbml::Matrix input = tbml::Matrix({ {
		cartPosition,
		cartAcceleration,
		poleAngle,
		poleAcceleration
	} });
	tbml::Matrix output = this->geneticData->propogate(input);
	float ft = output.get(0, 0) > 0.5f ? force : -force;

	// Update cart
	cartAcceleration = (ft + poleMass * poleLength * (poleVelocity * poleVelocity * sin(poleAngle) - poleAcceleration * cos(poleAngle))) / (cartMass + poleMass);
	poleAcceleration = g * (sin(poleAngle) + cos(poleAngle) * (-ft - poleMass * poleLength * poleVelocity * poleVelocity * sin(poleAngle)) / (cartMass + poleMass)) / (poleLength * (4.0f / 3.0f - (poleMass * cos(poleAngle) * cos(poleAngle)) / (cartMass + poleMass)));

	// Update pole
	cartPosition = cartPosition + cartVelocity * timeStep;
	poleAngle = poleAngle + poleVelocity * timeStep;
	cartVelocity += cartAcceleration * timeStep;
	poleVelocity += poleAcceleration * timeStep;
	this->time += timeStep;

	// Check finish conditions
	bool done = false;
	done |= abs(poleAngle) > angleLimit;
	done |= abs(cartPosition) > trackLimit;
	done |= time > timeLimit;
	if (done) {
		calculateFitness();
		instanceFinished = true;
		this->cartShape.setOutlineColor(sf::Color(100, 100, 140, 10));
		this->poleShape.setOutlineColor(sf::Color(100, 100, 140, 10));
	}
	return instanceFinished;
}

void NeuralPoleBalancerGI::render(sf::RenderWindow* window)
{
	// Update shape positions and rotations
	this->cartShape.setPosition(700.0f + cartPosition * METRE_TO_UNIT, 700.0f);
	this->poleShape.setPosition(700.0f + cartPosition * METRE_TO_UNIT, 700.0f);
	this->poleShape.setRotation(poleAngle * (180.0f / 3.141592653589793238463f));

	// Draw both to screen
	window->draw(this->cartShape);
	window->draw(this->poleShape);
}


float NeuralPoleBalancerGI::calculateFitness()
{
	// Dont calculate once finished
	if (this->instanceFinished) return this->instanceFitness;

	// Update and return
	this->instanceFitness = this->time;
	return this->instanceFitness;
}

#pragma endregion


#pragma region - NeuralPoleBalancerGI

NeuralPoleBalancerGS::NeuralPoleBalancerGS(
	float cartMass, float poleMass, float poleLength, float force,
	float trackLimit, float angleLimit, float timeLimit,
	std::vector<size_t> dataLayerSizes)
	:	cartMass(cartMass), poleMass(poleMass), poleLength(poleLength), force(force),
		trackLimit(trackLimit), angleLimit(angleLimit), timeLimit(timeLimit),
		dataLayerSizes(dataLayerSizes)
{}


NeuralGD* NeuralPoleBalancerGS::createData()
{
	// Create, randomize and return data
	NeuralGD* data = new NeuralGD(this->dataLayerSizes, tbml::sign);
	data->randomize();
	return data;
}

NeuralPoleBalancerGI* NeuralPoleBalancerGS::createInstance(NeuralGD* data)
{
	// Create and return instance
	NeuralPoleBalancerGI* inst = new NeuralPoleBalancerGI(
		cartMass, poleMass, poleLength, force,
		trackLimit, angleLimit, timeLimit,
		data);
	return inst;
}

#pragma endregion
