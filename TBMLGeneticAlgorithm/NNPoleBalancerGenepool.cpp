#include "stdafx.h"
#include "NNPoleBalancerGenepool.h"
#include "Utility.h"

#pragma region - NNPoleBalancerAgent

NNPoleBalancerAgent::NNPoleBalancerAgent(
	float cartMass, float poleMass, float poleLength, float force,
	float trackLimit, float angleLimit, float timeLimit,
	NNPoleBalancerAgent::GenomeCPtr&& genome)
	: Agent(std::move(genome)),
	cartMass(cartMass), poleMass(poleMass), poleLength(poleLength), force(force),
	trackLimit(trackLimit), angleLimit(angleLimit), timeLimit(timeLimit),
	netInput({ 1, 4 }, 0), network(this->genome->getNetwork()), poleAngle(0.1f)
{
	this->initVisual();
}

void NNPoleBalancerAgent::initVisual()
{
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

bool NNPoleBalancerAgent::step()
{
	if (this->isFinished) return true;

	// Calculate force with network
	netInput.setData({ 1, 4 }, {
		cartPosition,
		cartAcceleration,
		poleAngle,
		poleAcceleration });
	this->network.propogateMut(netInput);
	float ft = netInput(0, 0) > 0.5f ? force : -force;

	// Calculate acceleration
	cartAcceleration = (ft + poleMass * poleLength * (poleVelocity * poleVelocity * sin(poleAngle) - poleAcceleration * cos(poleAngle))) / (cartMass + poleMass);
	poleAcceleration = g * (sin(poleAngle) + cos(poleAngle) * (-ft - poleMass * poleLength * poleVelocity * poleVelocity * sin(poleAngle)) / (cartMass + poleMass)) / (poleLength * (4.0f / 3.0f - (poleMass * cos(poleAngle) * cos(poleAngle)) / (cartMass + poleMass)));

	// Update dynamics
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
	if (done)
	{
		calculateFitness();
		isFinished = true;
		this->cartShape.setOutlineColor(sf::Color(100, 100, 140, 10));
		this->poleShape.setOutlineColor(sf::Color(100, 100, 140, 10));
	}
	return isFinished;
}

void NNPoleBalancerAgent::render(sf::RenderWindow* window)
{
	// Update shape positions and rotations
	this->cartShape.setPosition(700.0f + cartPosition * METRE_TO_UNIT, 700.0f);
	this->poleShape.setPosition(700.0f + cartPosition * METRE_TO_UNIT, 700.0f);
	this->poleShape.setRotation(poleAngle * (180.0f / 3.141592653f));

	// Draw both to screen
	window->draw(this->cartShape);
	window->draw(this->poleShape);
}

float NNPoleBalancerAgent::calculateFitness()
{
	if (this->isFinished) return this->fitness;

	// Update and return
	this->fitness = this->time;
	return this->fitness;
}

#pragma endregion

#pragma region - NNPoleBalancerGenepool

NNPoleBalancerGenepool::NNPoleBalancerGenepool(
	float cartMass, float poleMass, float poleLength, float force,
	float trackLimit, float angleLimit, float timeLimit,
	std::function<GenomeCPtr(void)> createGenomeFn)
	: cartMass(cartMass), poleMass(poleMass), poleLength(poleLength), force(force),
	trackLimit(trackLimit), angleLimit(angleLimit), timeLimit(timeLimit),
	createGenomeFn(createGenomeFn)
{}

NNPoleBalancerGenepool::GenomeCPtr NNPoleBalancerGenepool::createGenome() const
{
	return createGenomeFn();
};

NNPoleBalancerGenepool::AgentPtr NNPoleBalancerGenepool::createAgent(NNPoleBalancerGenepool::GenomeCPtr&& data) const
{
	return std::make_unique<NNPoleBalancerAgent>(
		cartMass, poleMass, poleLength, force,
		trackLimit, angleLimit, timeLimit,
		std::move(data));
};

#pragma endregion
