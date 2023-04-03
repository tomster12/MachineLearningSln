
#pragma once

#include "GenepoolSimulation.h"
#include "CommonGeneticDatas.h"


class VectorListTargetGS;
class VectorListTargetGI : public tbml::GeneticInstance<VectorListGD>
{
private:
	VectorListTargetGS* sim;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	sf::Vector2f pos;
	float moveAcc;
	float radius;
	int currentIndex;

public:
	VectorListTargetGI(VectorListGD* geneticData) : GeneticInstance(geneticData), sim(nullptr), moveAcc(0), radius(0), currentIndex(-1) {};
	VectorListTargetGI(VectorListTargetGS* sim, sf::Vector2f startPos, float radius, float moveAcc, VectorListGD* geneticData);
	void initVisual();

	bool step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();
};


class VectorListTargetGS : public tbml::GenepoolSimulation<VectorListGD, VectorListTargetGI>
{
protected:
	sf::CircleShape target;
	sf::Vector2f instanceStartPos;
	float instanceRadius = 0.0f;
	sf::Vector2f targetPos;
	float targetRadius = 0.0f;
	float instancemoveAcc = 0.0f;
	int dataSize = 0;

	VectorListGD* createData() override;
	VectorListTargetGI* createInstance(VectorListGD* data) override;

public:
	VectorListTargetGS() {};
	VectorListTargetGS(
		sf::Vector2f instanceStartPos, float instanceRadius,
		float instancemoveAcc, int dataSize,
		sf::Vector2f targetPos, float targetRadius);

	void render(sf::RenderWindow* window) override;

	sf::Vector2f getTargetPos();
	float getTargetRadius();
};
