#include "stdafx.h"
#include "NNDriver.h"
#include "CommonImpl.h"
#include "Tensor.h"

NNDriverAgent::NNDriverAgent(
	NNDriverAgent::GenomeCPtr&& genome, const NNDriverGenepool* genepool,
	sf::Vector2f startPos, float maxDrivingSpeed, float drivingAcc, float steeringAcc, float moveDrag, float eyeLength, int maxIterations)
	: Agent(std::move(genome)), genepool(genepool),
	pos(startPos), maxDrivingSpeed(maxDrivingSpeed), drivingAcc(drivingAcc), steeringAcc(steeringAcc), moveDrag(moveDrag), eyeLength(eyeLength),
	maxIterations(maxIterations)
{
	this->initVisual();
}

void NNDriverAgent::initVisual()
{
	// Set up body shape
	bodyShape.setSize(sf::Vector2f(40.0f, 20.0f));
	bodyShape.setOrigin(20.0f, 10.0f);
	bodyShape.setOutlineColor(sf::Color::White);
	bodyShape.setOutlineThickness(1.0f);
	bodyShape.setFillColor(sf::Color::Transparent);
	bodyShape.setPosition(pos);
	bodyShape.setRotation(drivingAngle * (180.0f / 3.14159265f));

	// Set up eye shapes
	for (int i = 0; i < 5; i++)
	{
		float angle = drivingAngle + (i - 2) * 0.2f * 3.14159265f;

		sf::RectangleShape eyeShape;
		eyeShape.setSize(sf::Vector2f(eyeLength, 2.0f));
		eyeShape.setOrigin(0.0f, 1.0f);
		eyeShape.setPosition(pos);
		eyeShape.setRotation(angle * (180.0f / 3.14159265f));
		eyeShape.setOutlineColor(sf::Color::Green);
		eyeShape.setOutlineThickness(1.0f);
		eyeShape.setFillColor(sf::Color::Transparent);
		eyeShapes.push_back(eyeShape);
	}
}

bool NNDriverAgent::evaluate()
{
	if (isFinished) return true;

	// Stop if colliding
	if (genepool->isColliding(bodyShape))
	{
		isFinished = true;
		this->calculateFitness();
		bodyShape.setOutlineColor(sf::Color::Red);
		for (auto& eyeShape : eyeShapes) eyeShape.setOutlineColor(sf::Color::Transparent);
		return true;
	}

	// Raycast with eye shapes (-45, -20, 0, 20, 45)
	for (int i = 0; i < 5; i++)
	{
		float angle = drivingAngle + (i - 2) * 0.2f * 3.14159265f;
		eyeShapes[i].setPosition(pos);
		eyeShapes[i].setRotation(angle * (180.0f / 3.14159265f));
		eyeHits[i] = genepool->isColliding(eyeShapes[i]) ? 1.0f : 0.0f;
		eyeShapes[i].setOutlineColor(eyeHits[i] > 0.5f ? sf::Color::Red : sf::Color::Green);
	}

	// Calculate with brain
	float targetDir = genepool->getTargetDir(pos);
	netInput.setData({ 1, 8 }, { eyeHits[0], eyeHits[1], eyeHits[2], eyeHits[3], eyeHits[4], drivingSpeed, drivingAngle, targetDir });
	genome->getNetwork().propogateMut(netInput);

	// Update position, angle, speed
	drivingAngle += netInput(0, 0) * steeringAcc;
	drivingAngle = std::fmod(drivingAngle + 2.0f * 3.14159265f, 2.0f * 3.14159265f);
	drivingSpeed += netInput(0, 1) * drivingAcc;
	drivingSpeed = std::max(0.0f, std::min(maxDrivingSpeed, drivingSpeed));
	drivingSpeed *= moveDrag;
	pos.x += std::cos(drivingAngle) * drivingSpeed * (1.0f / 60.0f);
	pos.y += std::sin(drivingAngle) * drivingSpeed * (1.0f / 60.0f);
	currentIteration++;

	// Update shape
	bodyShape.setPosition(pos);
	bodyShape.setRotation(drivingAngle * (180.0f / 3.14159265f));

	// Check finish conditions
	if (currentIteration == maxIterations)
	{
		isFinished = true;
		this->calculateFitness();
		bodyShape.setOutlineColor(sf::Color::Yellow);
		for (auto& eyeShape : eyeShapes) eyeShape.setOutlineColor(sf::Color::Transparent);
	}
	else if (genepool->getTargetDist(pos) < 10.0f)
	{
		isFinished = true;
		hasReachedTarget = true;
		this->calculateFitness();
		bodyShape.setOutlineColor(sf::Color::Green);
		for (auto& eyeShape : eyeShapes) eyeShape.setOutlineColor(sf::Color::Transparent);
	}

	return isFinished;
}

void NNDriverAgent::render(sf::RenderWindow* window)
{
	// Draw shape
	window->draw(bodyShape);

	// Draw eyes
	for (const auto& eyeShape : eyeShapes) window->draw(eyeShape);
}

void NNDriverAgent::calculateFitness()
{
	// Reward reaching, punish dying, reward quickness
	fitness = 0.0f;
	if (hasReachedTarget)
	{
		fitness = 10.0f;
		fitness += 20.0f * (1.0f - (float)currentIteration / (float)maxIterations);
	}
	else
	{
		float dist = genepool->getTargetDist(pos);
		fitness = std::max(10.0f - dist / 10.0f, 0.0f);
	}
}

NNDriverGenepool::NNDriverGenepool(
	std::function<GenomeCnPtr(void)> createGenomeFn, std::function<AgentPtr(GenomeCnPtr)> createAgentFn,
	sf::Vector2f targetPos, float targetRadius, std::vector<sf::RectangleShape> worldShapes)
	: Genepool(createGenomeFn, createAgentFn),
	targetPos(targetPos), targetRadius(targetRadius), worldShapes(worldShapes)
{
	this->initVisual();
}

void NNDriverGenepool::initVisual()
{
	// Set up target shape
	targetShape.setRadius(targetRadius);
	targetShape.setOrigin(targetRadius, targetRadius);
	targetShape.setOutlineColor(sf::Color::Green);
	targetShape.setPosition(targetPos);
}

void NNDriverGenepool::render(sf::RenderWindow* window)
{
	Genepool::render(window);
	if (!this->showVisuals) return;

	// Draw target
	window->draw(targetShape);

	// Draw world shapes
	for (const auto& shape : worldShapes) window->draw(shape);
}

bool NNDriverGenepool::isColliding(sf::RectangleShape& shape) const
{
	for (const auto& worldShape : worldShapes)
	{
		if (isColliding(shape, worldShape)) return true;
	}
	return false;
}

float NNDriverGenepool::getTargetDist(sf::Vector2f pos) const
{
	float dx = targetPos.x - pos.x;
	float dy = targetPos.y - pos.y;
	return sqrt(dx * dx + dy * dy) - targetRadius;
}

float NNDriverGenepool::getTargetDir(sf::Vector2f pos) const
{
	float dx = targetPos.x - pos.x;
	float dy = targetPos.y - pos.y;
	return atan2(dy, dx);
}

bool NNDriverGenepool::isColliding(const sf::RectangleShape& shape1, const sf::RectangleShape& shape2)
{
	// Check exact collision between two rectangles with rotation
	sf::FloatRect rect1 = shape1.getGlobalBounds();
	sf::FloatRect rect2 = shape2.getGlobalBounds();
	return rect1.intersects(rect2);
}
