
#pragma once

#include "GenepoolSimulation.h"


class VectorListGD : public tbml::GeneticData<VectorListGD> {

private:
	std::vector<sf::Vector2f> values;
	int dataSize = 0;


public:
	VectorListGD() {}
	VectorListGD(int dataSize);
	VectorListGD(std::vector<sf::Vector2f> values);


	std::vector<sf::Vector2f>& getValues();
	sf::Vector2f getValue(int index);
	size_t getSize();


	void randomize() override;
	void mutate(float chance) override;
	VectorListGD* crossover(VectorListGD* otherData) override;
};


class VectorListTargetGS;
class VectorListTargetGI : public tbml::GeneticInstance<VectorListGD> {

private:
	VectorListTargetGS* sim;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	float moveSpeed;
	float radius;
	sf::Vector2f pos;
	int currentIndex;


public:
	VectorListTargetGI(VectorListGD* geneticData) : GeneticInstance(geneticData) {};
	VectorListTargetGI(VectorListTargetGS* sim, sf::Vector2f startPos, float radius, float moveSpeed, VectorListGD* geneticData);
	void initVisual();

	void step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

	bool getInstanceFinished() override;
	float getInstanceFitness() override;
};


class VectorListTargetGS : public tbml::GenepoolSimulation<VectorListGD, VectorListTargetGI> {

protected:
	sf::CircleShape target;
	sf::Vector2f instanceStartPos;
	float instanceRadius;
	sf::Vector2f targetPos;
	float targetRadius;
	float instanceMoveSpeed;
	int dataSize;

	VectorListGD* createData() override;
	VectorListTargetGI* createInstance(VectorListGD* data) override;

public:
	VectorListTargetGS() {};
	VectorListTargetGS(
		sf::Vector2f instanceStartPos, float instanceRadius,
		float instanceMoveSpeed, int dataSize,
		sf::Vector2f targetPos, float targetRadius);

	void render(sf::RenderWindow* window) override;

	sf::Vector2f getTargetPos();
	float getTargetRadius();
};
