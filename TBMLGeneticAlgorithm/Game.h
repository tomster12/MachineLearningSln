
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

	std::unique_ptr<tbml::GenepoolSimulationController> genepoolController;
	std::unique_ptr<UIManager> uiManager;

	void initialize();
	void update();
	void render();
};
