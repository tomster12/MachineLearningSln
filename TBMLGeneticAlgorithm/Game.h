#pragma once

#include "UIManager.h"
#include "GenepoolSimulation.h"

class GenepoolController
{
public:
	GenepoolController() {}

	GenepoolController(tbml::ga::IGenepoolPtr&& genepool)
		: genepool(std::move(genepool))
	{}

	void update()
	{
		if (!this->genepool->getGenepoolInitialized()) throw std::runtime_error("tbml::GenepoolSimulationController: Cannot update because uninitialized.");

		if (!this->genepool->getGenerationEvaluated())
		{
			if (this->toEvaluate) this->genepool->evaluateGeneration(!this->toFullEvaluate);
		}
		if (this->genepool->getGenerationEvaluated())
		{
			if (this->toAutoIterate) this->genepool->iterateGeneration();
		}
	};

	void render(sf::RenderWindow* window)
	{
		if (!this->genepool->getGenepoolInitialized()) throw std::runtime_error("tbml::GenepoolSimulation: Cannot render because uninitialized.");
		this->genepool->render(window);
	};

	void iterateGeneration()
	{
		if (!this->genepool->getGenepoolInitialized()) throw std::runtime_error("tbml::GenepoolSimulationController: Cannot iterateGeneration because uninitialized.");
		if (!this->genepool->getGenerationEvaluated()) return;
		this->genepool->iterateGeneration();
	}

	void setEvaluate(bool toEvaluate) { this->toEvaluate = toEvaluate; }

	void setFullEvaluate(bool toFullEvaluate) { this->toFullEvaluate = toFullEvaluate; }

	void setAutoIterate(bool toAutoIterate) { this->toAutoIterate = toAutoIterate; }

	void setShowVisuals(bool showVisuals) { this->genepool->setShowVisuals(showVisuals); }

	tbml::ga::IGenepoolPtr getGenepool() const { return this->genepool; }

protected:
	tbml::ga::IGenepoolPtr genepool = nullptr;
	bool toEvaluate = false;
	bool toFullEvaluate = false;
	bool toAutoIterate = false;
};

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

	std::unique_ptr<GenepoolController> genepoolController;
	std::unique_ptr<UIManager> uiManager;

	void initialize();
	void update();
	void render();
};
