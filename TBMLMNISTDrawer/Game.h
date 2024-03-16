#pragma once
#include "NeuralNetwork.h"
#include "DrawableGrid.h"

class Game
{
private:
	sf::RenderWindow* window;
	sf::Event sfEvent;
	sf::Clock dtClock;
	float dt;

	void initVariables();
	void update();
	void render();
	void updateGuess();

	tbml::nn::NeuralNetwork network;
	DrawableGrid grid;
	sf::Font font;
	sf::Text guessText;
	std::vector<sf::RectangleShape> guessChances;

public:
	Game();
	~Game();
	void run();
};
