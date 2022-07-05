
#pragma once

#include "UIManager.h"
#include "VectorGenepoolSimulation.h"


class Game {

private:
	sf::RenderWindow* window;
	sf::Event sfEvent;
	sf::Clock dtClock;
	float dt;

	VectorGenepoolSimulation genepool;
	UIManager uiManager;

	void initVariables();
	void update();
	void render();


public:
	Game();
	~Game();

	void run();
};
