
#include "stdafx.h"
#include "Game.h"


#pragma region Setup

Game::Game() {
    // Run initialization
    this->initVariables();
}


void Game::initVariables() {
    // Initialize variables
    this->window = NULL;
    this->dt = 0.0f;

    // Setup default options
    sf::VideoMode windowMode = sf::VideoMode::getDesktopMode();
    windowMode.width = 800;
    windowMode.height = 800;
    std::string title = "SFML Template";
    bool fullscreen = false;
    unsigned framerateLimit = 120;
    bool verticalSyncEnabled = false;

    // Setup window using default settings
    if (fullscreen) this->window = new sf::RenderWindow(windowMode, title, sf::Style::Fullscreen);
    else this->window = new sf::RenderWindow(windowMode, title, sf::Style::Titlebar | sf::Style::Close);
    this->window->setFramerateLimit(framerateLimit);
    this->window->setVerticalSyncEnabled(verticalSyncEnabled);

    // Init circle  
    this->testCircle = sf::CircleShape(100);
    this->testCircle.setFillColor(sf::Color(200, 0, 50));
}


Game::~Game() {
    // Delete window
    delete this->window;
}

#pragma endregion


#pragma region Main

void Game::run() {
    // Run game loop
    while (this->window->isOpen()) {
        this->update();
        this->render();
    }
}


void Game::update() {
	// Update delta time using dtClock
	this->dt = this->dtClock.restart().asSeconds();

    // Poll Events
    while (this->window->pollEvent(this->sfEvent)) {
        switch (this->sfEvent.type) {

        // Closed window
        case sf::Event::Closed:
            this->window->close();
            break;
        }
    }
}


void Game::render() {
    // Reset window background
    window->clear();

    // Render circle on mouse
    sf::Vector2i pos = sf::Mouse::getPosition(*this->window);
    this->testCircle.setPosition(pos.x - 100.0f, pos.y - 100.0f);
    window->draw(this->testCircle);

    // Display current draw window
    window->display();
}

#pragma endregion
