
#pragma once

#include "stdafx.h"
#include "Utility.h"
#include "ThreadPool.h"

namespace tbml
{
	template<class Data>
	class GeneticData
	{
	protected:
		using DataPtr = std::shared_ptr<const Data>;
		GeneticData() = default;
		~GeneticData() = default;

	public:
		GeneticData(const GeneticData&) = delete;
		GeneticData& operator=(const GeneticData&) = delete;
		GeneticData(const GeneticData&&) = delete;
		GeneticData& operator=(const GeneticData&&) = delete;

		virtual DataPtr crossover(const DataPtr& otherData, float mutateChance = 0.0f) const = 0;
	};

	template<class Data> // Data: GeneticData<Data>
	class GeneticInstance
	{
	protected:
		using DataPtr = std::shared_ptr<const Data>;

		const DataPtr geneticData;
		bool instanceFinished = false;
		float instanceFitness = 0;

	public:
		GeneticInstance(DataPtr&& geneticData) : geneticData(std::move(geneticData)), instanceFinished(false), instanceFitness(0.0f) {};

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

		virtual int getGenerationNumber() const = 0;
		virtual float getBestFitness() const = 0;
		virtual bool getInitialized() const = 0;
		virtual bool getGenerationEvaluated() const = 0;
	};

	using IGenepoolSimulationPtr = std::unique_ptr<IGenepoolSimulation>;

	template<class Data, class Inst> // Data: GeneticData<Data>, Inst: GeneticInstance<Data>
	class GenepoolSimulation : public IGenepoolSimulation
	{
	protected:
		using DataPtr = std::shared_ptr<const Data>;
		using InstPtr = std::unique_ptr<Inst>;

		bool enableMultithreadedStepEvaluation = false;
		bool enableMultithreadedFullEvaluation = false;
		bool multithreadSyncSteps = false;

		bool isInitialized = false;
		int populationSize = 0;
		float mutationRate = 0.0f;

		ThreadPool threadPool;
		int generationNumber = 0;
		int generationStepNumber = 0;
		std::vector<InstPtr> currentGeneration;
		bool isGenerationEvaluated = false;
		DataPtr bestData = nullptr;
		float bestFitness = 0.0f;

		virtual DataPtr createData() const { return std::make_shared<Data>(); }

		virtual InstPtr createInstance(DataPtr&& data) const { return std::make_unique<Inst>(std::move(data)); }

	public:
		GenepoolSimulation(bool enableMultithreadedStepEvaluation = false, bool enableMultithreadedFullEvaluation = true, bool multithreadSyncSteps = false)
		{
			if (enableMultithreadedFullEvaluation && enableMultithreadedStepEvaluation)
				throw std::runtime_error("tbml::GenepoolSimulation: Cannot have both enableMultithreadedFullEvaluation and enableMultithreadedStepEvaluation.");
			if (multithreadSyncSteps && !enableMultithreadedFullEvaluation)
				throw std::runtime_error("tbml::GenepoolSimulation: Cannot have multithreadSyncSteps without enableMultithreadedFullEvaluation.");
			this->enableMultithreadedStepEvaluation = enableMultithreadedStepEvaluation;
			this->enableMultithreadedFullEvaluation = enableMultithreadedFullEvaluation;
			this->multithreadSyncSteps = multithreadSyncSteps;
		}

		void render(sf::RenderWindow* window)
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot render because uninitialized.");

			for (const auto& inst : currentGeneration) inst->render(window);
		};

		void resetGenepool(int populationSize, float mutationRate)
		{
			// [INITIALIZATION] Initialize new instances
			this->currentGeneration.clear();
			for (int i = 0; i < populationSize; i++)
			{
				DataPtr geneticData = createData();
				InstPtr geneticInstance = createInstance(std::move(geneticData));
				this->currentGeneration.push_back(std::move(geneticInstance));
			}

			this->isInitialized = true;
			this->populationSize = populationSize;
			this->mutationRate = mutationRate;

			this->generationNumber = 1;
			this->generationStepNumber = 0;
			this->isGenerationEvaluated = false;

			initGeneration();
		};

		void initGeneration() {}

		void evaluateGeneration(bool step)
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot evaluateGeneration because uninitialized.");
			if (this->isGenerationEvaluated) return;

			// Helper function captures generation
			auto evalulateSubset = [&](bool step, int start, int end)
			{
				bool allFinished;
				do
				{
					allFinished = true;
					for (int i = start; i < end; i++) allFinished &= this->currentGeneration[i]->step();
				} while (!step && !allFinished);
				return allFinished;
			};

			std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
			bool allFinished = true;

			// Process generation (multi-threaded)
			if ((this->enableMultithreadedStepEvaluation && step) || (this->enableMultithreadedFullEvaluation && !step))
			{
				size_t threadCount = static_cast<size_t>(std::min(static_cast<int>(threadPool.size()), this->populationSize));
				std::vector<std::future<bool>> threadResults(threadCount);
				int subsetSize = static_cast<int>(ceil((float)this->populationSize / threadCount));
				do
				{
					for (size_t i = 0; i < threadCount; i++)
					{
						int startIndex = i * subsetSize;
						int endIndex = static_cast<int>(std::min(startIndex + subsetSize, this->populationSize));
						threadResults[i] = this->threadPool.enqueue([=] { return evalulateSubset(step || multithreadSyncSteps, startIndex, endIndex); });
					}

					for (auto&& result : threadResults) allFinished &= result.get();
					this->generationStepNumber++;

				} while (!step && !allFinished);
			}

			// Process generation (single-threaded)
			else
			{
				allFinished = evalulateSubset(step, 0, this->currentGeneration.size());
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

			// Sort generation and get best data
			std::sort(this->currentGeneration.begin(), this->currentGeneration.end(), [this](const auto& a, const auto& b) { return a->getFitness() > b->getFitness(); });
			const InstPtr& bestInstance = this->currentGeneration[0];
			this->bestData = DataPtr(bestInstance->getData());
			this->bestFitness = bestInstance->getFitness();
			std::cout << "Generation: " << this->generationNumber << ", best fitness: " << this->bestFitness << std::endl;

			// Create next generation with new instance of best data
			std::vector<InstPtr> nextGeneration;
			nextGeneration.push_back(createInstance(std::move(DataPtr(this->bestData))));

			// Selection helper function to pick a parent
			int selectAmount = static_cast<int>(ceil(this->currentGeneration.size() / 2.0f));
			auto transformFitness = [](float f) { return f * f; };
			float totalFitness = 0.0f;
			for (int i = 0; i < selectAmount; i++) totalFitness += transformFitness(this->currentGeneration[i]->getFitness());
			const auto& pickWeightedParent = [&]()
			{
				float r = getRandomFloat() * totalFitness;
				float cumSum = 0.0f;
				for (int i = 0; i < selectAmount; i++)
				{
					cumSum += transformFitness(this->currentGeneration[i]->getFitness());
					if (r <= cumSum) return this->currentGeneration[i]->getData();
				}
				return this->currentGeneration[selectAmount - 1]->getData();
			};

			for (int i = 0; i < this->populationSize - 1; i++)
			{
				// [SELECTION] Pick 2 parents from previous generation
				const DataPtr& parentDataA = pickWeightedParent();
				const DataPtr& parentDataB = pickWeightedParent();

				// [CROSSOVER], [MUTATION] Crossover and mutate new child data
				DataPtr childData = parentDataA->crossover(parentDataB, this->mutationRate);
				nextGeneration.push_back(createInstance(std::move(childData)));
			}

			// Set to new generation and update variables
			this->currentGeneration = std::move(nextGeneration);
			this->generationNumber++;
			this->isGenerationEvaluated = false;
			initGeneration();
		};

		int getGenerationNumber() const { return this->generationNumber; }

		float getBestFitness() const { return this->bestFitness; }

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

		bool getStepping() const { return this->isRunning; }

		bool getAutoStep() const { return this->autoStepEvaluate; }

		bool getAutoFinish() const { return this->autoIterate; }

		bool getAutoProcess() const { return this->autoFullEvaluate; }

		const IGenepoolSimulationPtr& getGenepool() const { return this->genepool; }

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
