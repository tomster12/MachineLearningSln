
#pragma once

#include "stdafx.h"
#include "UtilityFunctions.h"
#include "ThreadPool.h"

namespace tbml
{
	template<class Data>
	class GeneticData
	{
	protected:
		using DataPtr = std::shared_ptr<const Data>;

	public:
		virtual DataPtr crossover(const DataPtr& otherData, float mutateChance = 0.0f) const = 0;
	};

	template<class Data> // Data: GeneticData<Data>
	class GeneticInstance
	{
	protected:
		using DataPtr = std::shared_ptr<const Data>;

		const DataPtr geneticData = nullptr;
		bool instanceFinished = false;
		float instanceFitness = 0;

	public:
		GeneticInstance(const DataPtr&& geneticData) : geneticData(geneticData), instanceFinished(false), instanceFitness(0.0f) {};

		virtual bool step() = 0;
		virtual void render(sf::RenderWindow* window) = 0;

		const DataPtr& getData() const { return this->geneticData; };
		bool getFinished() const { return this->instanceFinished; };
		float getFitness() const { return this->instanceFitness; };
	};

	class IGenepoolSimulation
	{
	public:
		virtual void render(sf::RenderWindow* window) = 0;

		virtual void resetGenepool(int populationSize, float mutationRate) = 0;
		virtual void initGeneration() = 0;
		virtual void evaluateGeneration(bool step = false) = 0;
		virtual void iterateGeneration() = 0;

		virtual bool getInitialized() const = 0;
		virtual bool getGenerationEvaluated() const = 0;
	};

	using IGenepoolSimulationPtr = std::unique_ptr<IGenepoolSimulation>;

	template<class Data, class Inst> // Data: GeneticData<Data>, Inst: GeneticInstance<Data>
	class GenepoolSimulation : public IGenepoolSimulation
	{
	protected:
		using DataPtr = std::shared_ptr<const Data>;
		using InstPtr = std::shared_ptr<Inst>;

		bool enableMultithreading = false;
		bool linkMultithreadedSteps = false;

		bool isInitialized = false;
		int populationSize = 0;
		float mutationRate = 0.0f;

		int generationNumber = 0;
		int generationStepNumber = 0;
		std::vector<InstPtr> currentGeneration;
		bool isGenerationEvaluated = false;
		DataPtr bestCurrentData = nullptr;
		float bestCurrentFitness = 0.0f;
		ThreadPool threadPool;

		virtual DataPtr createData() { return std::make_shared<Data>(); }

		virtual InstPtr createInstance(const DataPtr&& data) { return std::make_shared<Inst>(std::move(data)); }

	public:
		GenepoolSimulation(bool enableMultithreading = false, bool linkMultithreadedSteps = false)
		{
			this->enableMultithreading = enableMultithreading;
			this->linkMultithreadedSteps = linkMultithreadedSteps;
		}

		void render(sf::RenderWindow* window)
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot render because uninitialized.");

			for (auto inst : currentGeneration) inst->render(window);
		};

		void resetGenepool(int populationSize, float mutationRate)
		{
			// [INITIALIZATION] Initialize new instances
			this->currentGeneration.clear();
			for (int i = 0; i < populationSize; i++)
			{
				DataPtr geneticData = createData();
				InstPtr geneticInstance = createInstance(std::move(geneticData));
				this->currentGeneration.push_back(geneticInstance);
			}

			this->isInitialized = true;
			this->populationSize = populationSize;
			this->mutationRate = mutationRate;

			this->generationNumber = 1;
			this->generationStepNumber = 0;
			this->isGenerationEvaluated = false;
			this->bestCurrentData = nullptr;
			this->bestCurrentFitness = 0.0f;

			initGeneration();
		};

		void initGeneration() {}

		void evaluateGeneration(bool step)
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot evaluateGeneration because uninitialized.");
			if (this->isGenerationEvaluated) return;

			// Helper functions
			auto stepSubset = [&](const std::vector<InstPtr>& subset)
			{
				bool allFinished = true;
				for (const auto& inst : subset) allFinished &= inst->step();
				return allFinished;
			};
			auto processSubset = [&](const std::vector<InstPtr>& subset)
			{
				for (bool allFinished = false; !allFinished;)
				{
					allFinished = true;
					for (const auto& inst : subset) allFinished &= inst->step();
				}
				return true;
			};

			std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
			bool allFinished = true;

			// Process generation (multi-threaded)
			if (this->enableMultithreading)
			{
				size_t threadCount = static_cast<size_t>(std::min(static_cast<int>(threadPool.size()), this->populationSize));
				std::vector<std::future<bool>> threadResults(threadCount);
				int threadGenerationSubsetCount = static_cast<int>(ceil((float)this->populationSize / threadCount));

				do
				{
					for (size_t i = 0; i < threadCount; i++)
					{
						int startIndex = i * threadGenerationSubsetCount;
						int endIndex = static_cast<int>(std::min(startIndex + threadGenerationSubsetCount, this->populationSize));
						const std::vector<InstPtr> threadGenerationSubset(this->currentGeneration.begin() + startIndex, this->currentGeneration.begin() + endIndex);

						if (this->linkMultithreadedSteps) threadResults[i] = threadPool.enqueue([&] { return stepSubset(threadGenerationSubset); });
						else threadResults[i] = threadPool.enqueue([&] { return processSubset(threadGenerationSubset); });
					}

					for (auto&& result : threadResults) allFinished &= result.get();
					this->generationStepNumber++;

				} while (!step && !allFinished);
			}

			// Process generation (single-threaded)
			else
			{
				if (step) allFinished = stepSubset(this->currentGeneration);
				else allFinished = processSubset(this->currentGeneration);
				this->generationStepNumber++;
			}

			std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
			auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
			// std::cout << "Processed generation: " << us.count() / 1000.0f << "ms" << std::endl;
			this->generationStepNumber++;
			if (allFinished) this->isGenerationEvaluated = true;
		}

		void iterateGeneration()
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot iterateGeneration because uninitialized.");
			if (!this->isGenerationEvaluated) return;

			// [SELECTION] Sort, then cull the bottom half of the generation
			std::sort(this->currentGeneration.begin(), this->currentGeneration.end(), [this](auto a, auto b) { return a->getFitness() < b->getFitness(); });
			int selectAmount = static_cast<int>(ceil(this->currentGeneration.size() / 2.0f));
			std::vector<InstPtr> selectedInstances(this->currentGeneration.end() - selectAmount, this->currentGeneration.end());

			// get best instance
			InstPtr bestInstance = selectedInstances[selectAmount - 1];
			std::cout << "Generation: " << this->generationNumber << ", best fitness: " << bestInstance->getFitness() << std::endl;

			// setup parent selection function
			auto transformFitness = [](float f) { return f * f; };
			float totalFitness = 0.0f;
			for (const auto& inst : selectedInstances) totalFitness += (transformFitness(inst->getFitness()));
			const auto& pickRandomData = [selectedInstances, totalFitness, transformFitness]()
			{
				float r = getRandomFloat() * totalFitness;
				float cumSum = 0.0f;
				for (const auto& inst : selectedInstances)
				{
					cumSum += transformFitness(inst->getFitness());
					if (r <= cumSum) return inst->getData();
				}
				return selectedInstances[selectedInstances.size() - 1]->getData();
			};

			// Create the next generation
			std::vector<InstPtr> newInstances(this->populationSize);
			for (int i = 0; i < this->populationSize - 1; i++)
			{
				// [CROSSOVER], [MUTATION] Create new genetic data
				const DataPtr& dataA = pickRandomData();
				const DataPtr& dataB = pickRandomData();
				DataPtr newData = dataA->crossover(dataB, this->mutationRate);
				newInstances[i] = createInstance(std::move(newData));
			}

			// Keep the best data
			newInstances[this->populationSize - 1] = createInstance(std::move(bestInstance->getData()));

			// Set to new generation and update variables
			this->currentGeneration = newInstances;
			this->generationNumber++;
			this->isGenerationEvaluated = false;
			initGeneration();
		};

		bool getInitialized() const { return this->isInitialized; }

		bool getGenerationEvaluated() const { return this->isGenerationEvaluated; }
	};

	class GenepoolSimulationController
	{
	protected:
		IGenepoolSimulationPtr genepool = nullptr;

		bool isRunning = false;
		bool autoStepEvaluate = false;
		bool autoFullEvaluate = false;
		bool autoIterate = false;

	public:
		GenepoolSimulationController() {}

		GenepoolSimulationController(IGenepoolSimulationPtr genepool)
			: genepool(std::move(genepool))
		{}

		void update()
		{
			if (!this->genepool->getInitialized()) throw std::runtime_error("tbml::GenepoolSimulationController: Cannot update because uninitialized.");
			if (this->genepool->getGenerationEvaluated() || !this->isRunning) return;

			this->evaluateGeneration(!this->autoFullEvaluate);
		};

		void render(sf::RenderWindow* window)
		{
			if (!this->genepool->getInitialized()) throw std::runtime_error("tbml::GenepoolSimulation: Cannot render because uninitialized.");

			this->genepool->render(window);
		};

		void evaluateGeneration(bool step = false)
		{
			if (!this->genepool->getInitialized()) throw std::runtime_error("tbml::GenepoolSimulationController: Cannot evaluateGeneration because uninitialized.");
			if (this->genepool->getGenerationEvaluated()) return;

			this->genepool->evaluateGeneration(step);
			if (this->autoIterate && this->genepool->getGenerationEvaluated()) this->iterateGeneration();
		}

		void iterateGeneration()
		{
			if (!this->genepool->getInitialized()) throw std::runtime_error("tbml::GenepoolSimulationController: Cannot iterateGeneration because uninitialized.");
			if (!this->genepool->getGenerationEvaluated()) return;

			this->genepool->iterateGeneration();
			this->setRunning(this->autoStepEvaluate || this->autoFullEvaluate);
		}

		bool getStepping() const { return this->isRunning; };

		bool getAutoStep() const { return this->autoStepEvaluate; };

		bool getAutoFinish() const { return this->autoIterate; };

		bool getAutoProcess() const { return this->autoFullEvaluate; };

		void setRunning(bool isRunning) { this->isRunning = isRunning; }

		void setAutoStepEvaluate(bool autoStepEvaluate)
		{
			this->autoStepEvaluate = autoStepEvaluate;
			if (!this->isRunning && this->autoStepEvaluate) this->setRunning(true);
		};

		void setAutoFullEvaluate(bool autoFullEvaluate)
		{
			this->autoFullEvaluate = autoFullEvaluate;
			this->setRunning(this->autoStepEvaluate || this->autoFullEvaluate);
		}

		void setAutoIterate(bool autoIterate)
		{
			this->autoIterate = autoIterate;
			if (this->autoIterate && this->genepool->getGenerationEvaluated()) this->iterateGeneration();
		}
	};
}
