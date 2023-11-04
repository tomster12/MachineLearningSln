
#include "stdafx.h"
#include "Game.h"

#include "UtilityFunctions.h"
#include "GenepoolSimulation.h"

#include "VectorListTargetGS.h"
//#include "NeuralTargetGS.h"
//#include "NeuralIceTargetsGS.h"
//#include "NeuralPoleBalancerGS.h"

Game::Game()
	: window(NULL), dt(0)
{
	this->initialize();
}

Game::~Game()
{
	delete this->window;
}

void Game::initialize()
{
	srand(0);

	// Initialize window
	sf::VideoMode windowMode = sf::VideoMode::getDesktopMode();
	windowMode.width = 1400;
	windowMode.height = 1000;
	std::string title = "Genetic Algorithm";
	bool fullscreen = false;
	unsigned framerateLimit = 60;
	bool verticalSyncEnabled = false;

	if (fullscreen) this->window = new sf::RenderWindow(windowMode, title, sf::Style::Fullscreen);
	else this->window = new sf::RenderWindow(windowMode, title, sf::Style::Titlebar | sf::Style::Close);
	this->window->setFramerateLimit(framerateLimit);
	this->window->setVerticalSyncEnabled(verticalSyncEnabled);

	// Initialize genepool object
	tbml::IGenepoolSimulationPtr genepool(new VectorListTargetGS(
		{ 700.0f, 600.0f }, 4.0f, 4.0f, 500, { 700.0f, 100.0f }, 20.0f));
	//tbml::IGenepoolSimulationPtr genepool(new NeuralTargetGS(
	//	{ 700.0f, 850.0f }, 2.0f, 2.0f, 1000, { 2, 2 }, 20.0f, { 700.0f, 150.0f }, 500.0f));
	//tbml::IGenepoolSimulationPtr genepool(new NeuralIceTargetsGS(
	//	{ 700.0f, 850.0f }, 300.0f, 3000, { 4, 4, 2 }, { { 150.0f, 150.0f }, { 920.0f, 400.0f }, { 300.0f, 850.0f }, { 550.0f, std::unique_ptr<320.0f } }, 20.0f));
	//tbml::IGenepoolSimulationPtr genepool(new NeuralPoleBalancerGS(
	//	1.0f, 0.1f, 0.5f, 2.0f, 0.6f, 0.25f, 20.0f, { 4, 1 }, tbml::tanh));

	genepool->resetGenepool(2000, 0.05f);
	this->genepoolController = std::make_unique<tbml::GenepoolSimulationController>(std::move(genepool));

	// Initialize UI
	this->uiManager = std::make_unique<UIManager>();
	float sp = 6.0f;
	float sz = 30.0f;
	std::vector<UIElement*> elements = {
		new UIButton(this->window, { sp + 0 * (sp + sz), sp + 0 * (sp + sz) }, { sz, sz }, "assets/startstepping.png", [this]() { this->genepoolController->setRunning(true); }),
		new UIButton(this->window, { sp + 1 * (sp + sz), sp + 0 * (sp + sz) }, { sz, sz }, "assets/pausestepping.png", [this]() { this->genepoolController->setRunning(false); }),
		new UIButton(this->window, { sp + 2 * (sp + sz), sp + 0 * (sp + sz) }, { sz, sz }, "assets/process.png", [this]() { this->genepoolController->evaluateGeneration(); }),
		new UIButton(this->window, { sp + 3 * (sp + sz), sp + 0 * (sp + sz) }, { sz, sz }, "assets/finish.png", [this]() { this->genepoolController->iterateGeneration(); }),
		new UIToggleButton(this->window, { sp + 4 * (sp + sz), sp + 0 * (sp + sz) }, { sz, sz }, "assets/hide.png", false, [this](bool toggled) { global::showVisuals = !toggled; }),
		new UIToggleButton(this->window, { sp + 0 * (sp + sz), sp + 1 * (sp + sz) }, { sz, sz }, "assets/autostep.png", false, [this](bool toggled) { this->genepoolController->setAutoStepEvaluate(toggled); }),
		new UIToggleButton(this->window, { sp + 2 * (sp + sz), sp + 1 * (sp + sz) }, { sz, sz }, "assets/autoprocess.png", false, [this](bool toggled) { this->genepoolController->setAutoFullEvaluate(toggled); }),
		new UIToggleButton(this->window, { sp + 3 * (sp + sz), sp + 1 * (sp + sz) }, { sz, sz }, "assets/autofinish.png", false, [this](bool toggled) { this->genepoolController->setAutoIterate(toggled); })
	};
	for (auto e : elements) this->uiManager->addElement(e);
}

void Game::run()
{
	while (this->window->isOpen())
	{
		this->update();
		this->render();
	}
}

void Game::update()
{
	this->dt = this->dtClock.restart().asSeconds();

	while (this->window->pollEvent(this->sfEvent))
	{
		switch (this->sfEvent.type)
		{

		case sf::Event::Closed:
			this->window->close();
			break;
		}
	}

	this->genepoolController->update();
	this->uiManager->update();
}

void Game::render()
{
	window->clear();

	if (global::showVisuals) this->genepoolController->render(window);
	this->uiManager->render(window);

	window->display();
}
