
#pragma once


class Game {

private:
	// Private variables
	sf::RenderWindow* window;
	sf::Event sfEvent;
	sf::Clock dtClock;
	float dt;

	sf::CircleShape testCircle;

	// Private functions
	void initVariables();
	void update();
	void render();

public:
	// Public constructors
	Game();
	~Game();

	// Public functions
	void run();
};