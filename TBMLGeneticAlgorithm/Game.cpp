#include "Tensor.h"
#include "stdafx.h"
#include "Game.h"
#include "Utility.h"
#include "GenepoolSimulation.h"
#include "VectorListTarget.h"
#include "NNTarget.h"
#include "NNPoleBalancer.h"
#include "NNDriver.h"

#define GENEPOOL_TYPE 2

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

#if GENEPOOL_TYPE == 0

	VectorListTargetGenepool* genepool = new VectorListTargetGenepool(
		[]() { return std::make_shared<VectorListGenome>(500); },
		nullptr, sf::Vector2f{ 700.0f, 100.0f }, 20.0f);

	genepool->setCreateAgentFn(
		[=](VectorListTargetGenepool::GenomeCnPtr data) { return std::make_unique<VectorListTargetAgent>(std::move(data), genepool, sf::Vector2f{ 700.0f, 600.0f }, 4.0f, 4.0f); });

#elif GENEPOOL_TYPE == 1

	NNTargetGenepool* genepool = new NNTargetGenepool(
		[]() { return std::make_shared<NNGenome>(tbml::nn::NeuralNetwork({ std::make_shared<tbml::nn::Layer::Dense>(4, 2), std::make_shared<tbml::nn::Layer::TanH>() })); },
		nullptr, { { 300.0f, 150.0f }, { 1100.0f, 400.0f }, { 450.0f, 850.0f }, { 700.0f, 320.0f } }, 4.0f);

	genepool->setCreateAgentFn(
		[=](NNTargetGenepool::GenomeCnPtr data) { return std::make_unique<NNTargetAgent>(std::move(data), genepool, sf::Vector2f{ 700.0f, 850.0f }, 2.0f, 400.0f, 0.99f, 3000); });

#elif GENEPOOL_TYPE == 2

	auto genepool = new tbml::ga::Genepool<NNGenome, NNPoleBalancerAgent>(
		[]() { return std::make_shared<NNGenome>(tbml::nn::NeuralNetwork({ std::make_shared<tbml::nn::Layer::Dense>(4, 1), std::make_shared<tbml::nn::Layer::TanH>() })); },
		[](std::shared_ptr<const NNGenome> genome) { return std::make_unique<NNPoleBalancerAgent>(std::move(genome), 1.0f, 0.1f, 0.7f, 1.0f, 1.0f, 0.4f, 20.0f); });

#elif GENEPOOL_TYPE == 3

	float PI = 3.14159265359f;
	std::vector<Body> worldBodies;
	worldBodies.push_back(Body({ 250.0f, 550.0f }, { 50.0f, 850.0f }, PI * 0.12f));
	worldBodies.push_back(Body({ 650.0f, 600.0f }, { 50.0f, 500.0f }, PI * 0.12f));
	worldBodies.push_back(Body({ 750.0f, 150.0f }, { 700.0f, 50.0f }));
	worldBodies.push_back(Body({ 800.0f, 550.0f }, { 400.0f, 50.0f }, PI * 0.4f));
	worldBodies.push_back(Body({ 1200.0f, 480.0f }, { 700.0f, 50.0f }, PI * 0.4f));

	std::vector<sf::Vector2f> targets;
	targets.push_back({ 580.0f, 265.0f });
	targets.push_back({ 970.0f, 265.0f });
	targets.push_back({ 1030.0f, 700.0f });
	targets.push_back({ 550.0f, 930.0f });
	targets.push_back({ 580.0f, 265.0f });
	targets.push_back({ 970.0f, 265.0f });
	targets.push_back({ 1030.0f, 700.0f });
	targets.push_back({ 550.0f, 930.0f });

	auto genepool = new NNDriverGenepool(
		[]()
	{
		return std::make_shared<NNGenome>(tbml::nn::NeuralNetwork({
			std::make_shared<tbml::nn::Layer::Dense>(8, 5),
			std::make_shared<tbml::nn::Layer::ReLU>(),
			std::make_shared<tbml::nn::Layer::Dense>(5, 2),
			std::make_shared <tbml::nn::Layer::TanH>() }));
	},
		nullptr, targets, 40.0f, worldBodies);

	genepool->setCreateAgentFn(
		[=](NNDriverGenepool::GenomeCnPtr data)
	{
		return std::make_unique<NNDriverAgent>(
			std::move(data), genepool,
			sf::Vector2f{ 380.0f, 780.0f }, 500.0f, 20.0f, 0.3f, 0.98f, 120.0f, 300);
	});

#endif

	// Reset genepool generation, initialize controller
	genepool->configThreading(false, true, false);
	genepool->resetGenepool(1000, 0.05f);
	this->genepoolController = std::make_unique<GenepoolController>(tbml::ga::IGenepoolPtr(genepool));

	// Initialize UI using spacing constants
	this->uiManager = std::make_unique<UIManager>();
	float osp = 6.0f;
	float sp = 6.0f;
	float sz = 30.0f;
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIToggleButton(
		this->window, { osp + sp + 0 * (sp + sz), osp + sp + 0 * (sp + sz) }, { sz, sz }, "assets/autoEvaluate.png", false, [&](bool toggled) { this->genepoolController->setEvaluate(toggled); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIToggleButton(
		this->window, { osp + sp + 0 * (sp + sz), osp + sp + 1 * (sp + sz) }, { sz, sz }, "assets/autoFullEvaluate.png", false, [&](bool toggled) { this->genepoolController->setFullEvaluate(toggled); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIButton(
		this->window, { osp + sp + 1 * (sp + sz), osp + sp + 0 * (sp + sz) }, { sz, sz }, "assets/iterate.png", [&]() { this->genepoolController->iterateGeneration(); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIToggleButton(
		this->window, { osp + sp + 1 * (sp + sz), osp + sp + 1 * (sp + sz) }, { sz, sz }, "assets/autoIterate.png", false, [&](bool toggled) { this->genepoolController->setAutoIterate(toggled); })));
	this->uiManager->addElement(std::shared_ptr<UIElement>(new UIToggleButton(
		this->window, { osp + sp + 2 * (sp + sz), osp + sp + 0 * (sp + sz) }, { sz, sz }, "assets/show.png", true, [&](bool toggled) { this->genepoolController->setShowVisuals(toggled); })));
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

	this->genepoolController->render(window);
	this->uiManager->render(window);

	window->display();
}
