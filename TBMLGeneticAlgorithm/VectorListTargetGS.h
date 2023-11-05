
#pragma once

#include "GenepoolSimulation.h"
#include "CommonImpl.h"

class VectorListTargetGS;
class VectorListTargetGI : public tbml::GeneticInstance<VectorListGD>
{
public:
	VectorListTargetGI(VectorListTargetGI::DataPtr&& geneticData) : GeneticInstance(std::move(geneticData)) {};
	VectorListTargetGI(VectorListTargetGI::DataPtr&& geneticData, const VectorListTargetGS* sim, sf::Vector2f startPos, float radius, float moveAcc);
	void initVisual();

	bool step() override;
	void render(sf::RenderWindow* window) override;

	float calculateDist();
	float calculateFitness();

private:
	const VectorListTargetGS* sim = nullptr;
	sf::CircleShape shape;

	sf::Vector2f startPos;
	sf::Vector2f pos;
	float moveAcc = 0.0f;
	float radius = 0.0f;
	int currentIndex = -1;
};

class VectorListTargetGS : public tbml::GenepoolSimulation<VectorListGD, VectorListTargetGI>
{
public:
	VectorListTargetGS() {};
	VectorListTargetGS(
		sf::Vector2f instanceStartPos, float instanceRadius,
		float instancemoveAcc, int dataSize,
		sf::Vector2f targetPos, float targetRadius);

	void render(sf::RenderWindow* window) override;

	sf::Vector2f getTargetPos() const;
	float getTargetRadius() const;

protected:
	sf::CircleShape target;
	sf::Vector2f instanceStartPos;
	float instanceRadius = 0.0f;
	sf::Vector2f targetPos;
	float targetRadius = 0.0f;
	float instancemoveAcc = 0.0f;
	int dataSize = 0;

	DataPtr createData() const override;
	InstPtr createInstance(DataPtr&& data) const override;
};
