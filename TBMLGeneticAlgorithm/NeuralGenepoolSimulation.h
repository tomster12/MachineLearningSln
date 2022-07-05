
//#pragma once
//
//#include "GenepoolSimulation.h"
//#include "NeuralNetwork.h"
//#include "Matrix.h"
//
//
//class NeuralData : public GeneticData<NeuralData> {
//
//private:
//	NeuralNetwork network;
//
//
//public:
//	NeuralData();
//	NeuralData(std::vector<int> layerSizes_);
//
//	Matrix propogate(Matrix input);
//
//	void randomize() override;
//	void mutate(float chance) override;
//	NeuralData* crossover(NeuralData* otherData) override;
//};
//
//
//class NeuralGenepoolSimulation;
//class NeuralInstance : public GeneticInstance<NeuralData> {
//
//private:
//	NeuralGenepoolSimulation* sim;
//	sf::CircleShape shape;
//
//	sf::Vector2f startPos;
//	float radius;
//	float moveSpeed;
//	int iterations;
//	sf::Vector2f pos;
//	int currentIteration;
//
//
//public:
//	NeuralInstance(NeuralData* data) : GeneticInstance(data) {};
//	NeuralInstance(NeuralGenepoolSimulation* sim, sf::Vector2f startPos, float radius, float moveSpeed, int iterations, NeuralData* data);
//	void initVisual();
//
//	void step() override;
//	void render(sf::RenderWindow* window) override;
//
//	float calculateDist();
//	float calculateFitness();
//
//	bool getInstanceFinished() override;
//	float getInstanceFitness() override;
//};
//
//
//class NeuralGenepoolSimulation : public GenepoolSimulation<NeuralData, NeuralInstance> {
//
//protected:
//	sf::CircleShape target;
//	sf::Vector2f instanceStartPos;
//	float instanceRadius;
//	float instanceMoveSpeed;
//	int instanceIterations;
//	sf::Vector2f targetPos;
//	float targetRadius;
//	std::vector<int> dataLayerSizes;
//
//	NeuralData* createData() override;
//	NeuralInstance* createInstance(NeuralData* data) override;
//
//public:
//	NeuralGenepoolSimulation() {};
//	NeuralGenepoolSimulation(
//		sf::Vector2f instanceStartPos, float instanceRadius,
//		float instanceMoveSpeed, int instanceIterations,
//		float targetRadius, std::vector<int> dataLayerSizes);
//
//	void render(sf::RenderWindow* window) override;
//
//	void initGeneration() override;
//
//	sf::Vector2f getTargetPos();
//	float getTargetRadius();
//};
