#include "stdafx.h"
#include "Game.h"
#include "Utility.h"
#include "GenepoolSimulation.h"
#include "VectorListTargetGenepool.h"
#include "NNTargetGenepool.h"
#include "NNIceTargetsGenepool.h"
#include "NNPoleBalancerGenepool.h"

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
	/*
	tbml::ga::IGenepoolPtr genepool(new VectorListTargetGenepool(
		{ 700.0f, 600.0f }, 4.0f, 4.0f,
		{ 700.0f, 100.0f }, 20.0f,
		500
	));
	*/

	/*
	tbml::ga::IGenepoolPtr genepool(new NNTargetGenepool(
		{ 700.0f, 850.0f }, 2.0f, 2.0f, 1000,
		20.0f, { 700.0f, 150.0f }, 500.0f,
		[]()
	{
		return std::make_shared<NNGenome>(std::make_shared<tbml::fn::SquareError>(),
		std::vector<std::shared_ptr<tbml::nn::Layer>>{ std::make_shared<tbml::nn::DenseLayer>(2, 2, std::make_shared<tbml::fn::TanH>()) });
	}));
	*/

	/*
	tbml::ga::IGenepoolPtr genepool(new NNIceTargetsGenepool(
		{ 700.0f, 850.0f }, 2.0f, 400.0f, 0.99f, 3000,
		{ { 300.0f, 150.0f }, { 1100.0f, 400.0f }, { 450.0f, 850.0f }, { 700.0f, 320.0f } }, 4.0f,
		[]()
	{
		return std::make_shared<NNGenome>(std::make_shared<tbml::fn::SquareError>(),
		std::vector<std::shared_ptr<tbml::nn::Layer>>{ std::make_shared<tbml::nn::DenseLayer>(6, 4, std::make_shared<tbml::fn::TanH>()), std::make_shared<tbml::nn::DenseLayer>(6, 2, std::make_shared<tbml::fn::TanH>())
	});
	}));
	*/

	tbml::ga::IGenepoolPtr genepool(new NNPoleBalancerGenepool(
		1.0f, 0.1f, 0.5f, 2.0f,
		0.6f, 0.25f, 20.0f,
		[]()
	{
		return std::make_shared<NNGenome>(std::make_shared<tbml::fn::SquareError>(),
		std::vector<std::shared_ptr<tbml::nn::Layer>>{ std::make_shared<tbml::nn::DenseLayer>(4, 1, std::make_shared<tbml::fn::TanH>()) });
	}));

	genepool->resetGenepool(2000, 0.05f);

	this->genepoolController = std::make_unique<tbml::ga::GenepoolController>(std::move(genepool));

	// Initialize UI
	this->uiManager = std::make_unique<UIManager>();
	float osp = 6.0f;
	float sp = 6.0f;
	float sz = 30.0f;

	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIButton(
		this->window, { osp + sp + 0 * (sp + sz), osp + sp + 0 * (sp + sz) }, { sz, sz }, "assets/startStepping.png", [&]() { this->genepoolController->setRunning(true); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIButton(
		this->window, { osp + sp + 1 * (sp + sz), osp + sp + 0 * (sp + sz) }, { sz, sz }, "assets/stopStepping.png", [&]() { this->genepoolController->setRunning(false); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIButton(
		this->window, { osp + sp + 2 * (sp + sz), osp + sp + 0 * (sp + sz) }, { sz, sz }, "assets/evaluate.png", [&]() { this->genepoolController->evaluateGeneration(); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIButton(
		this->window, { osp + sp + 3 * (sp + sz), osp + sp + 0 * (sp + sz) }, { sz, sz }, "assets/iterate.png", [&]() { this->genepoolController->iterateGeneration(); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIToggleButton(
		this->window, { osp + sp + 4 * (sp + sz), osp + sp + 0 * (sp + sz) }, { sz, sz }, "assets/show.png", false, [&](bool toggled) { global::showVisuals = !toggled; })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIToggleButton(
		this->window, { osp + sp + 0 * (sp + sz), osp + sp + 1 * (sp + sz) }, { sz, sz }, "assets/autoStartStepping.png", false, [&](bool toggled) { this->genepoolController->setAutoStepEvaluate(toggled); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIToggleButton(
		this->window, { osp + sp + 2 * (sp + sz), osp + sp + 1 * (sp + sz) }, { sz, sz }, "assets/autoEvaluate.png", false, [&](bool toggled) { this->genepoolController->setAutoFullEvaluate(toggled); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIToggleButton(
		this->window, { osp + sp + 3 * (sp + sz), osp + sp + 1 * (sp + sz) }, { sz, sz }, "assets/autoIterate.png", false, [&](bool toggled) { this->genepoolController->setAutoIterate(toggled); })));

	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIDynamicText(
		this->window, { osp + sp * 1.2f, osp + sp + osp + 2 * (sp + sz) + 0 }, 15, [&]() { return std::string("Generation: ") + std::to_string(this->genepoolController->getGenepool()->getGenerationNumber()); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIDynamicText(
		this->window, { osp + sp * 1.2f, osp + sp + osp + 2 * (sp + sz) + 20 }, 15, [&]() { return std::string("Evaluated: ") + std::string(this->genepoolController->getGenepool()->getGenerationEvaluated() ? "True" : "False"); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIDynamicText(
		this->window, { osp + sp * 1.2f, osp + sp + osp + 2 * (sp + sz) + 40 }, 15, [&]() { return std::string("Best Fitness: ") + std::to_string(this->genepoolController->getGenepool()->getBestFitness()); })));
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
