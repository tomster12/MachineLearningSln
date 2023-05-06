
#include "stdafx.h"
#include "Game.h"

#include "UtilityFunctions.h"
#include "GenepoolSimulation.h"

#include "VectorListTargetGS.h"
#include "NeuralTargetGS.h"
#include "NeuralIceTargetsGS.h"
#include "NeuralPoleBalancerGS.h"


Game::Game()
{
	this->initVariables();
}

Game::~Game()
{
	delete this->window;
	delete this->genepool;
}


void Game::initVariables()
{
	// Initialize variables
	this->window = NULL;
	this->dt = 0.0f;
	srand(0);

	// Setup default options
	sf::VideoMode windowMode = sf::VideoMode::getDesktopMode();
	windowMode.width = 1400;
	windowMode.height = 1000;
	std::string title = "Genetic Algorithm";
	bool fullscreen = false;
	unsigned framerateLimit = 60;
	bool verticalSyncEnabled = false;

	// Setup window using default settings
	if (fullscreen) this->window = new sf::RenderWindow(windowMode, title, sf::Style::Fullscreen);
	else this->window = new sf::RenderWindow(windowMode, title, sf::Style::Titlebar | sf::Style::Close);
	this->window->setFramerateLimit(framerateLimit);
	this->window->setVerticalSyncEnabled(verticalSyncEnabled);

	// Setup genepool structure
	//this->genepool = new VectorListTargetGS({ 700.0f, 600.0f }, 4.0f, 4.0f, 500, { 700.0f, 100.0f }, 20.0f);
	//this->genepool = new NeuralTargetGS({ 700.0f, 850.0f }, 2.0f, 2.0f, 1000, { 2, 2 }, 20.0f, { 700.0f, 150.0f }, 500.0f);
	//this->genepool = new NeuralIceTargetsGS({ 700.0f, 850.0f }, 300.0f, 3000, { 4, 4, 2 }, { { 150.0f, 150.0f }, { 920.0f, 400.0f }, { 300.0f, 850.0f }, { 550.0f, 320.0f } }, 20.0f);
	this->genepool = new NeuralPoleBalancerGS(1.0f, 0.1f, 0.5f, 2.0f, 0.6f, 0.25f, 20.0f, { 4, 1 }, tbml::tanh);

	// Restart the genepool
	this->genepool->restartGenepool(2000, 0.05f);

	// Initialize UI
	this->uiManager = UIManager();
	float spacing = 6.0f;
	float size = 30.0f;
	this->uiManager.addElement(new UIButton(this->window, { spacing + 0 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "assets/startstepping.png",
		[this]() { this->genepool->setStepping(true); }
	));
	this->uiManager.addElement(new UIButton(this->window, { spacing + 1 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "assets/pausestepping.png",
		[this]() { this->genepool->setStepping(false); }
	));
	this->uiManager.addElement(new UIButton(this->window, { spacing + 2 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "assets/process.png",
		[this]() { this->genepool->processGeneration(); }
	));
	this->uiManager.addElement(new UIButton(this->window, { spacing + 3 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "assets/finish.png",
		[this]() { this->genepool->finishGeneration(); }
	));
	this->uiManager.addElement(new UIToggleButton(this->window, { spacing + 4 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "assets/hide.png", false,
		[this](bool toggled) { global::showVisuals = !toggled; }
	));
	this->uiManager.addElement(new UIToggleButton(this->window, { spacing + 0 * (spacing + size), spacing + 1 * (spacing + size) }, { size, size }, "assets/autostep.png", false,
		[this](bool toggled) { this->genepool->setAutoStep(toggled); }
	));
	this->uiManager.addElement(new UIToggleButton(this->window, { spacing + 2 * (spacing + size), spacing + 1 * (spacing + size) }, { size, size }, "assets/autoprocess.png", false,
		[this](bool toggled) { this->genepool->setAutoProcess(toggled); }
	));
	this->uiManager.addElement(new UIToggleButton(this->window, { spacing + 3 * (spacing + size), spacing + 1 * (spacing + size) }, { size, size }, "assets/autofinish.png", false,
		[this](bool toggled) { this->genepool->setAutoFinish(toggled); }
	));
}

void Game::update()
{
	// Update delta time using dtClock
	this->dt = this->dtClock.restart().asSeconds();

	// Poll Events
	while (this->window->pollEvent(this->sfEvent))
	{
		switch (this->sfEvent.type)
		{

			// Closed window
		case sf::Event::Closed:
			this->window->close();
			break;
		}
	}

	// Update genepool
	this->genepool->update();
	this->uiManager.update();
}

void Game::render()
{
	// Reset window background
	window->clear();

	// Render genepool
	if (global::showVisuals) this->genepool->render(window);
	this->uiManager.render(window);

	// Display current draw window
	window->display();
}


void Game::run()
{
	// Run game loop
	while (this->window->isOpen())
	{
		this->update();
		this->render();
	}
}
