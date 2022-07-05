
#pragma once

#include "UIManager.h"
//#include "VectorListTargetGS.h"
//#include "NeuralTargetGS.h"
#include "NeuralRocketGS.h"


class Game {

private:
	sf::RenderWindow* window;
	sf::Event sfEvent;
	sf::Clock dtClock;
	float dt;

	//VectorListTargetGS genepool;
	//NeuralTargetGS genepool;
	NeuralRocketGS genepool;
	UIManager uiManager;

	void initVariables();
	void update();
	void render();


public:
	Game();
	~Game();

	void run();
};
