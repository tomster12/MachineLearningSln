
#pragma once

#include "UIManager.h"
#include "GenepoolSimulation.h"
#include "VectorListTargetGS.h"
#include "NeuralTargetGS.h"
#include "NeuralIceTargetsGS.h"

class Game
{
public:
	Game();
	~Game();

	void run();

private:
	sf::RenderWindow* window;
	sf::Event sfEvent;
	sf::Clock dtClock;
	float dt;

	tbml::IGenepoolSimulation* genepool;
	UIManager uiManager;

	void initVariables();
	void update();
	void render();
};
