#include "stdafx.h"
#include "NNDriver.h"
#include "CommonImpl.h"
#include "Tensor.h"

void Body::updateShape(sf::RectangleShape& shape) const
{
	shape.setPosition(pos);
	shape.setRotation(rot * (180.0f / 3.14159265f));
}

void Body::recalculateVertices()
{
	float c = std::cos(rot);
	float s = std::sin(rot);
	float w = size.x / 2.0f;
	float h = size.y / 2.0f;
	vertices = {
		{ pos.x + w * c - h * s, pos.y + w * s + h * c },
		{ pos.x - w * c - h * s, pos.y - w * s + h * c },
		{ pos.x - w * c + h * s, pos.y - w * s - h * c },
		{ pos.x + w * c + h * s, pos.y + w * s - h * c }
	};
}

std::pair<float, float> Body::projectVerticesOnAxis(const std::vector<sf::Vector2f>& vertices, const sf::Vector2f& axis)
{
	// Find min and max projection of all vertices on axis
	float min = (vertices[0].x * axis.x + vertices[0].y * axis.y);
	float max = min;
	for (const auto& vertex : vertices)
	{
		float projection = (vertex.x * axis.x + vertex.y * axis.y);
		if (projection < min) min = projection;
		if (projection > max) max = projection;
	}
	return { min, max };
}

bool Body::overlapOnAxis(const std::vector<sf::Vector2f>& vertices1, const std::vector<sf::Vector2f>& vertices2, const sf::Vector2f& axis)
{
	// Overlap on axis if projection min and max overlap
	auto pair1 = projectVerticesOnAxis(vertices1, axis);
	auto pair2 = projectVerticesOnAxis(vertices2, axis);
	return !(pair1.second < pair2.first || pair2.second < pair1.first);
}

bool Body::isColliding(const Body& other) const
{
	// Get all axes as all edges from both rectangles
	std::vector<sf::Vector2f> axes;
	for (size_t i = 0; i < 4; ++i)
	{
		sf::Vector2f edge1 = vertices[i] - vertices[(i + 1) % 4];
		sf::Vector2f edge2 = other.vertices[i] - other.vertices[(i + 1) % 4];
		axes.push_back(sf::Vector2f(-edge1.y, edge1.x));
		axes.push_back(sf::Vector2f(-edge2.y, edge2.x));
	}

	// Collision if there is no axis of separation
	for (const auto& axis : axes)
	{
		if (!overlapOnAxis(vertices, other.vertices, axis)) return false;
	}
	return true;
}

NNDriverAgent::NNDriverAgent(
	NNDriverAgent::GenomeCPtr&& genome, const NNDriverGenepool* genepool,
	sf::Vector2f startPos, float maxDrivingSpeed, float drivingAcc, float steeringSpeed, float moveDrag, float eyeLength, int maxIterations)
	: Agent(std::move(genome)), genepool(genepool),
	maxDrivingSpeed(maxDrivingSpeed), drivingAcc(drivingAcc), steeringSpeed(steeringSpeed), moveDrag(moveDrag), eyeLength(eyeLength), maxIterations(maxIterations)
{
	// Initialize bodies
	mainBody = Body(startPos, sf::Vector2f(40.0f, 20.0f), 1.5f * 3.14159265f);
	for (int i = 0; i < 5; i++)
	{
		float angle = mainBody.rot + (i - 2) * 0.2f * 3.14159265f;
		sf::Vector2f eyeStartPos = mainBody.pos
			+ sf::Vector2f(20.0f * std::cos(mainBody.rot), 20.0f * std::sin(mainBody.rot))
			+ sf::Vector2f(0.5f * eyeLength * std::cos(angle), 0.5f * eyeLength * std::sin(angle));
		eyeBodies.push_back(Body(eyeStartPos, sf::Vector2f(eyeLength, 3.0f), angle));
	}
}

void NNDriverAgent::initVisual()
{
	if (isVisualInit) return;

	// Initialize eye colors
	eyeColourHit = sf::Color(200, 100, 100, 120);
	eyeColourMiss = sf::Color(100, 100, 100, 120);

	// Set up body shape
	mainShape.setFillColor(sf::Color::Transparent);
	mainShape.setOutlineColor(sf::Color(255, 255, 255, 120));
	mainShape.setOutlineThickness(1.0f);
	mainShape.setSize(mainBody.size);
	mainShape.setOrigin(mainBody.size.x / 2.0f, mainBody.size.y / 2.0f);

	// Set up eye shapes
	for (int i = 0; i < 5; i++)
	{
		float angle = mainBody.rot + (i - 2) * 0.2f * 3.14159265f;
		sf::RectangleShape eyeShape;
		eyeShape.setFillColor(eyeColourMiss);
		eyeShape.setSize(eyeBodies[i].size);
		eyeShape.setOrigin(eyeBodies[i].size.x / 2.0f, eyeBodies[i].size.y / 2.0f);
		eyeShapes.push_back(eyeShape);
	}

	if (isFinished) this->setFinishedVisual();

	isVisualInit = true;
}

void NNDriverAgent::setFinishedVisual()
{
	if (!isVisualInit) return;

	// Set main shape based on finish type
	// 0: Collided, 1: Max iterations, 2: Reached target
	if (finishType == 0) mainShape.setOutlineColor(sf::Color(200, 100, 100, 60));
	else if (finishType == 1) mainShape.setOutlineColor(sf::Color(200, 200, 100, 60));
	else if (finishType == 2) mainShape.setOutlineColor(sf::Color(100, 200, 100, 60));

	// Set eye shapes to transparent
	for (auto& eyeShape : eyeShapes) eyeShape.setFillColor(sf::Color::Transparent);
}

bool NNDriverAgent::evaluate()
{
	if (isFinished) return true;

	// Finish 0: Collided with world
	mainBody.recalculateVertices();
	if (genepool->isColliding(mainBody))
	{
		isFinished = true;
		finishType = 0;
		this->calculateFitness();
		setFinishedVisual();
		return true;
	}

	// Raycast with eye shapes (-45, -20, 0, 20, 45)
	for (int i = 0; i < 5; i++)
	{
		float angle = mainBody.rot + (i - 2) * 0.2f * 3.14159265f;
		sf::Vector2f eyeStartPos = mainBody.pos
			+ sf::Vector2f(20.0f * std::cos(mainBody.rot), 20.0f * std::sin(mainBody.rot))
			+ sf::Vector2f(0.5f * eyeLength * std::cos(angle), 0.5f * eyeLength * std::sin(angle));
		eyeBodies[i].pos = eyeStartPos;
		eyeBodies[i].rot = angle;
		eyeBodies[i].recalculateVertices();
		eyeHits[i] = genepool->isColliding(eyeBodies[i]) ? 1.0f : 0.0f;
	}

	// Calculate with brain (bias, eyes, speed, angle, angle diff)
	float rotDiff = genepool->getTargetDir(mainBody.pos) - mainBody.rot;
	netInput.setData({ 1, 9 }, { 1.0f, eyeHits[0], eyeHits[1], eyeHits[2], eyeHits[3], eyeHits[4], drivingSpeed, mainBody.rot, rotDiff });
	genome->getNetwork().propogateMut(netInput);

	// Update position, angle, speed
	mainBody.rot += netInput(0, 0) * steeringSpeed;
	drivingSpeed += netInput(0, 1) * drivingAcc;
	mainBody.rot = std::fmod(mainBody.rot + 2.0f * 3.14159265f, 2.0f * 3.14159265f);
	drivingSpeed = std::max(0.0f, std::min(maxDrivingSpeed, drivingSpeed * moveDrag));
	mainBody.pos.x += std::cos(mainBody.rot) * drivingSpeed * (1.0f / 60.0f);
	mainBody.pos.y += std::sin(mainBody.rot) * drivingSpeed * (1.0f / 60.0f);
	currentIteration++;

	// Finish 1: Max iterations
	if (currentIteration == maxIterations)
	{
		isFinished = true;
		finishType = 1;
		this->calculateFitness();
		setFinishedVisual();
		return true;
	}

	// Finish 2: Reached target
	if (genepool->getTargetDist(mainBody.pos) < genepool->getTargetRadius())
	{
		isFinished = true;
		finishType = 2;
		this->calculateFitness();
		setFinishedVisual();
		return true;
	}

	return isFinished;
}

void NNDriverAgent::render(sf::RenderWindow* window)
{
	if (!isVisualInit) this->initVisual();

	// Update shapes
	mainBody.updateShape(mainShape);

	// Update eye shapes
	for (int i = 0; i < 5; i++)
	{
		if (!isFinished) eyeShapes[i].setFillColor(eyeHits[i] > 0.5f ? eyeColourHit : eyeColourMiss);
		eyeBodies[i].updateShape(eyeShapes[i]);
	}

	// Draw shapes
	window->draw(mainShape);
	for (const auto& eyeShape : eyeShapes) window->draw(eyeShape);
}

void NNDriverAgent::calculateFitness()
{
	// Punish collision with world with distance / 2
	if (finishType == 0)
	{
		float dist = genepool->getTargetDist(mainBody.pos);
		fitness = 5.0f / std::max(1.0f, dist / 20.0f);
	}

	// Punish distance from target if not reached
	else if (finishType == 1)
	{
		float dist = genepool->getTargetDist(mainBody.pos);
		fitness = 10.0f / std::max(1.0f, dist / 20.0f);
	}

	// Reward reaching target in as little time as possible
	else if (finishType == 2)
	{
		fitness = 10.0f + 20.0f / std::max(1.0f, (float)currentIteration / 20.0f);
	}
}

NNDriverGenepool::NNDriverGenepool(
	std::function<GenomeCnPtr(void)> createGenomeFn, std::function<AgentPtr(GenomeCnPtr)> createAgentFn,
	sf::Vector2f targetPos, float targetRadius, std::vector<Body> worldBodies)
	: Genepool(createGenomeFn, createAgentFn),
	targetPos(targetPos), targetRadius(targetRadius), worldBodies(worldBodies)
{
	this->initVisual();
}

void NNDriverGenepool::initVisual()
{
	// Set up target shape
	targetShape.setPosition(targetPos);
	targetShape.setFillColor(sf::Color::Transparent);
	targetShape.setOutlineColor(sf::Color::Green);
	targetShape.setOutlineThickness(1.0f);
	targetShape.setRadius(targetRadius);
	targetShape.setOrigin(targetRadius, targetRadius);

	// Set up world shapes
	for (auto& body : worldBodies)
	{
		sf::RectangleShape shape;
		shape.setFillColor(sf::Color::Transparent);
		shape.setOutlineColor(sf::Color::White);
		shape.setOutlineThickness(1.0f);
		shape.setSize(body.size);
		shape.setOrigin(body.size.x / 2.0f, body.size.y / 2.0f);
		body.updateShape(shape);
		body.recalculateVertices();
		worldShapes.push_back(shape);
	}
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

bool NNDriverGenepool::isColliding(Body& body) const
{
	for (const auto& other : worldBodies)
	{
		if (body.isColliding(other)) return true;
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
