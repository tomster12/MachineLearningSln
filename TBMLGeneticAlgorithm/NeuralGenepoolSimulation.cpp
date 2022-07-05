
#include "stdafx.h"
#include "NeuralGenepoolSimulation.h"
//#include "Globals.h"
//#include "Matrix.h"
//
//
//#pragma region - NeuralData
//
//NeuralData::NeuralData(std::vector<int> layerSizes)
//	: network(layerSizes) {}
//
//
//void NeuralData::randomize() {
//	// TODO
//};
//
//void NeuralData::mutate(float chance) {
//	// TODO
//};
//
//NeuralData* NeuralData::crossover(NeuralData* otherData) {
//	// TODO
//	return this;
//};
//
//#pragma endregion
//
//
//#pragma region - NeuralInstance
//
//NeuralInstance::NeuralInstance(NeuralGenepoolSimulation* sim, sf::Vector2f startPos, float radius, float moveSpeed, int iterations, NeuralData* data)
//	: GeneticInstance(data), sim(sim), pos(startPos), radius(radius), moveSpeed(moveSpeed), iterations(iterations), currentIteration(0) {
//
//	// Initialize variables
//	if (Globals::SHOW_VISUALS) initVisual();
//}
//
//void NeuralInstance::initVisual() {
//	// Initialize all visual variables
//	this->shape.setRadius(this->radius);
//	this->shape.setOrigin(this->radius, this->radius);
//	this->shape.setFillColor(sf::Color::Transparent);
//	this->shape.setOutlineColor(sf::Color::White);
//	this->shape.setOutlineThickness(1.0f);
//}
//
//
//void NeuralInstance::step() {
//	if (this->instanceFinished) return;
//
//	// Move position by current vector
//	sf::Vector2f targetPos = this->sim->getTargetPos();
//	Matrix input = Matrix({ { this->pos.x, this->pos.y, targetPos.x, targetPos.y } });
//	Matrix output = this->geneticData->propogate(input);
//	this->pos.x += (output.get(0, 0) * 2 - 1) * this->moveSpeed;
//	this->pos.y += (output.get(0, 1) * 2 - 1) * this->moveSpeed;
//	this->currentIteration++;
//
//	// Check finish conditions
//	float dist = calculateDist();
//	if (this->currentIteration == this->iterations || dist < 0.0f) {
//		this->calculateFitness();
//		this->instanceFinished = true;
//	}
//};
//
//void NeuralInstance::render(sf::RenderWindow* window) {
//	// Update shape position and colour
//	this->shape.setPosition(this->pos.x, this->pos.y);
//
//	// Draw shape to window
//	window->draw(this->shape);
//};
//
//
//float NeuralInstance::calculateDist() {
//	// Calculate distance to target
//	float dx = this->sim->getTargetPos().x - pos.x;
//	float dy = this->sim->getTargetPos().y - pos.y;
//	float fullDistSq = sqrt(dx * dx + dy * dy);
//	float radii = this->radius + this->sim->getTargetRadius();
//	return fullDistSq - radii;
//}
//
//float NeuralInstance::calculateFitness() {
//	// Dont calculate once finished
//	if (this->instanceFinished) return this->instanceFitness;
//
//	// Calculate fitness
//	float dist = calculateDist();
//	float fitness = 0.0f;
//
//	if (dist > 0.0f) {
//		fitness = 0.5f * (1.0f - dist / 500.0f);
//		fitness = fitness < 0.0f ? 0.0f : fitness;
//
//	} else {
//		// TODO
//		//float dataPct = static_cast<float>(this->currentIndex) / static_cast<float>(this->geneticData->getSize());
//		//fitness = 1.0f - 0.5f * dataPct;
//	}
//
//	// Update and return
//	this->instanceFitness = fitness;
//	return this->instanceFitness;
//};
//
//
//bool NeuralInstance::getInstanceFinished() { return this->instanceFinished; };
//
//float NeuralInstance::getInstanceFitness() { return calculateFitness(); };
//
//#pragma endregion
//
//
//#pragma region - NeuralGenepoolSimulation
//
//NeuralGenepoolSimulation::NeuralGenepoolSimulation(
//	sf::Vector2f instanceStartPos, float instanceRadius,
//	float instanceMoveSpeed, int instanceIterations,
//	float targetRadius, std::vector<int> dataLayerSizes)
//	: instanceStartPos(instanceStartPos), instanceRadius(instanceRadius),
//	instanceMoveSpeed(instanceMoveSpeed), instanceIterations(instanceIterations),
//	targetPos(targetPos), targetRadius(targetRadius),
//	dataLayerSizes(dataLayerSizes) {
//
//	// Initialize variables
//	this->target.setRadius(this->targetRadius);
//	this->target.setOrigin(this->targetRadius, this->targetRadius);
//	this->target.setFillColor(sf::Color::Transparent);
//	this->target.setOutlineColor(sf::Color::White);
//	this->target.setOutlineThickness(1.0f);
//	this->target.setPosition(this->targetPos);
//};
//
//
//NeuralData* NeuralGenepoolSimulation::createData() {
//	// Create, randomize and return data
//	NeuralData* data = new NeuralData(this->dataLayerSizes);
//	data->randomize();
//	return data;
//};
//
//NeuralInstance* NeuralGenepoolSimulation::createInstance(NeuralData* data) {
//	// Create and return instance
//	NeuralInstance* inst = new NeuralInstance(this, this->instanceStartPos, this->instanceRadius, this->instanceMoveSpeed, this->instanceIterations, data);
//	return inst;
//};
//
//
//void NeuralGenepoolSimulation::render(sf::RenderWindow* window) {
//	GenepoolSimulation::render(window);
//
//	// Draw target
//	window->draw(this->target);
//}
//
//
//void NeuralGenepoolSimulation::initGeneration() {
//	// Randomize target location
//	// TODO
//}
//
//
//sf::Vector2f NeuralGenepoolSimulation::getTargetPos() { return this->targetPos; }
//
//float NeuralGenepoolSimulation::getTargetRadius() { return this->targetRadius; }
//
//#pragma endregion
