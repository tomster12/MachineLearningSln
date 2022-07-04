
#pragma once

#include "GenepoolSimulation.h"


class Vector2fData : public GeneticData<Vector2fData> {

private:
	std::vector<sf::Vector2f> values;
	int dataSize = 0;


public:
	Vector2fData() {}
	Vector2fData(int dataSize);
	Vector2fData(std::vector<sf::Vector2f> values);


	std::vector<sf::Vector2f>& getValues();
	sf::Vector2f getValue(int index);
	size_t getSize();


	void randomize() override;
	void mutate(float chance) override;
	Vector2fData* crossover(Vector2fData* otherData) override;
};


class VectorGenepoolSimulation;
class VectorInstance : public GeneticInstance<Vector2fData> {

private:
	VectorGenepoolSimulation* sim;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	float moveSpeed;
	float radius;
	sf::Vector2f pos;
	int currentIndex;


public:
	VectorInstance(Vector2fData* data) : GeneticInstance(data) {};
	VectorInstance(VectorGenepoolSimulation* sim, sf::Vector2f startPos, float radius, float moveSpeed, Vector2fData* data);
	void initVisual();
	void resetInstance();

	void step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

	bool getInstanceFinished() override;
	float getInstanceFitness() override;
};


class VectorGenepoolSimulation : public GenepoolSimulation<Vector2fData, VectorInstance> {

protected:
	sf::CircleShape target;
	sf::Vector2f instanceStartPos;
	float instanceRadius;
	sf::Vector2f targetPos;
	float targetRadius;
	float instanceMoveSpeed;
	int dataSize;

	Vector2fData* createData() override;
	VectorInstance* createInstance(Vector2fData* data) override;

public:
	VectorGenepoolSimulation() {};
	VectorGenepoolSimulation(sf::Vector2f instanceStartPos, float instanceRadius, sf::Vector2f targetPos, float targetRadius, float instanceMoveSpeed, int dataSize);

	void render(sf::RenderWindow* window) override;

	sf::Vector2f getTargetPos();
	float getTargetRadius();
};
