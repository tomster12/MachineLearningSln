
#include "stdafx.h"
#include "Utility.h"
#include "GenepoolSimulation.h"


template<class D> GeneticInstance<D>::GeneticInstance(D* data) {
	// Initialize variables
	this->geneticData = data;
	resetInstance();
};

template<class D> void GeneticInstance<D>::resetInstance() {
	// Reset variables
	instanceFinished = false;
	instanceFitness = 0.0f;
}


template<class D, class I> D* GenepoolSimulation<D, I>::createData() {
	// Create, randomize and return data
	D* data = new D();
	data->randomize();
	return data;
}

template<class D, class I> I* GenepoolSimulation<D, I>::createInstance(D* data) {
	// Create and return instance
	I* inst = new I(data);
	return inst;
}


template<class D, class I> GenepoolSimulation<D, I>::~GenepoolSimulation() {
	if (!this->isInitialized) return;

	// Delete all data / instances
	for (int i = 0; i < this->generationCount; i++) {
		delete this->currentGeneration[i].data;
		delete this->currentGeneration[i].instance;
	}
}

template<class D, class I> void GenepoolSimulation<D, I>::initGenepool(int generationCount, float mutationRate) {
	// Initialize variables
	this->generationCount = generationCount;
	this->mutationRate = mutationRate;

	this->isInitialized = true;
	this->currentGeneration = std::vector<GeneticEvaluation>();
	this->generationNumber = 1;
	this->isRunning = false;
	this->isFinished = false;
	this->autoStart = false;
	this->autoFinish = false;
	this->autoComplete = false;

	// [INITIALIZATION] Initialize all data and instances
	for (int i = 0; i < this->generationCount; i++) {
		D* data = createData();
		I* inst = createInstance(data);
		this->currentGeneration.push_back({ data, inst, false, 0.0f });
	}
};

template<class D, class I> void GenepoolSimulation<D, I>::resetGeneration() {
	if (!this->isInitialized) return;

	// Render all current generation
	for (auto& eval : currentGeneration) {
		eval.instance->resetInstance();
		eval.fitness = 0.0f;
	}
	this->isRunning = false;
	this->isFinished = false;
	this->autoComplete = false;

	// Handle auto start
	if (this->autoStart) startGeneration();
};


template<class D, class I> void GenepoolSimulation<D, I>::update() {
	if (!this->isInitialized) return;
	if (this->isFinished || !this->isRunning) return;

	// Step / complete generation then check for auto finish
	if (this->autoComplete) completeGeneration();
	else stepGeneration();
	if (this->autoFinish && this->isFinished) finishGeneration();
};

template<class D, class I> void GenepoolSimulation<D, I>::render(sf::RenderWindow* window) {
	if (!this->isInitialized) return;

	// Render all current generation
	for (auto& eval : currentGeneration) eval.instance->render(window);
};


template<class D, class I> void GenepoolSimulation<D, I>::startGeneration() {
	// Start all current generation
	if (!this->isInitialized) return;
	if (this->isRunning || this->isFinished) return;
	this->isRunning = true;
};

template<class D, class I> void GenepoolSimulation<D, I>::pauseGeneration() {
	// Start all current generation
	if (!this->isInitialized) return;
	if (!this->isRunning || this->isFinished) return;
	this->isRunning = false;
};

template<class D, class I> void GenepoolSimulation<D, I>::stepGeneration() {
	if (!this->isInitialized) return;
	if (this->isFinished) return;

	// Step all unfinished instances
	bool allFinished = true;
	for (GeneticEvaluation& eval : currentGeneration) {
		if (!eval.isFinished) {
			eval.instance->step();

			// Cache final fitness once finished
			if (eval.instance->getInstanceFinished()) {
				eval.isFinished = true;
				eval.fitness = eval.instance->getInstanceFitness();
			} else allFinished = false;
		}
	}

	// Once all finished update variables
	if (allFinished) {
		this->isFinished = true;
		this->isRunning = false;
	}
}

template<class D, class I> void GenepoolSimulation<D, I>::completeGeneration() {
	if (!this->isInitialized) return;

	// Finish generation if needed
	if (this->isFinished) finishGeneration();

	// Step generation until finished
	while (!this->isFinished) stepGeneration();

	// Check for auto finish
	if (this->autoFinish) finishGeneration();
}

template<class D, class I> void GenepoolSimulation<D, I>::finishGeneration() {
	if (!this->isInitialized) return;
	if (this->isRunning || !this->isFinished) return;

	// [SELECTION] Sort, then cull the bottom half of the generation
	std::sort(this->currentGeneration.begin(), this->currentGeneration.end(), [this](auto a, auto b) { return a.fitness < b.fitness; });
	int amount = static_cast<int>(this->currentGeneration.size() / 2);
	std::vector<GeneticEvaluation> selectedGeneration(this->currentGeneration.begin() + amount, this->currentGeneration.end());
	std::cout << "Best Fitness: " << selectedGeneration[selectedGeneration.size() - 1].fitness << std::endl;

	// setup parent selection function
	float totalFitness = 0.0f;
	for (auto& eval : selectedGeneration) totalFitness += eval.fitness;
	auto pickParent = [selectedGeneration, totalFitness](float r) {
		float cumSum = 0.0f;
		if (r == 1.0f) return selectedGeneration[selectedGeneration.size() - 1].data;
		for (auto& eval : selectedGeneration) {
			cumSum += eval.fitness / totalFitness;
			if (r <= cumSum) return eval.data;
		}
	};

	// Create the next generation
	std::vector<GeneticEvaluation> newGeneration;
	for (int i = 0; i < this->generationCount; i++) {
		D* dataA = pickParent(getRandomFloat());
		D* dataB = pickParent(getRandomFloat());

		// [CROSSOVER], [MUTATION] Create new genetic data
		D* newData = dataA->crossover(dataB);
		newData->mutate(this->mutationRate);
		I* newInst = createInstance(newData);
		GeneticEvaluation newEval = { newData, newInst, false, 0.0f };
		newGeneration.push_back(newEval);
	}

	// Delete old, replace with new, update variables
	for (auto& eval : this->currentGeneration) { delete eval.data; delete eval.instance; }
	this->currentGeneration = newGeneration;
	this->generationNumber++;
	this->isFinished = false;

	// Handle auto start
	if (this->autoStart) startGeneration();
}


template<class D, class I> bool GenepoolSimulation<D, I>::getRunning() { return isRunning; };

template<class D, class I> bool GenepoolSimulation<D, I>::getFinished() { return isFinished; };

template<class D, class I> bool GenepoolSimulation<D, I>::getAutoStart() { return this->autoStart; };

template<class D, class I> bool GenepoolSimulation<D, I>::getAutoFinish() { return this->autoFinish; };

template<class D, class I> bool GenepoolSimulation<D, I>::getAutoComplete() { return this->autoComplete; };

template<class D, class I> void GenepoolSimulation<D, I>::setAutoStart(bool autoStart) { this->autoStart = autoStart; if (!this->isRunning && this->autoStart) this->startGeneration(); };

template<class D, class I> void GenepoolSimulation<D, I>::setAutoFinish(bool autoFinish) { this->autoFinish = autoFinish; if (this->isFinished && this->autoFinish) this->finishGeneration(); }

template<class D, class I> void GenepoolSimulation<D, I>::setAutoComplete(bool autoComplete) { this->autoComplete = autoComplete; if (this->autoComplete) this->completeGeneration(); }
