#pragma once

#include "Utility.h"
#include "ThreadPool.h"

// Require SFML to be imported
// TODO: Figure out if this is best way
namespace sf
{
	class RenderWindow;
}

namespace tbml
{
	namespace ga
	{
		template<class TGenome>
		class Genome
		{
		protected:
			using GenomeCPtr = std::shared_ptr<const TGenome>;
			Genome() = default;
			~Genome() = default;

		public:
			Genome(const Genome&) = delete;
			Genome& operator=(const Genome&) = delete;
			Genome(const Genome&&) = delete;
			Genome& operator=(const Genome&&) = delete;

			virtual GenomeCPtr crossover(const GenomeCPtr& otherGenome, float mutateChance = 0.0f) const = 0;
		};

		template<class TGenome> // TGenome: Genome<TGenome>
		class Agent
		{
		protected:
			using GenomeCPtr = std::shared_ptr<const TGenome>;
			const GenomeCPtr genome;
			bool isFinished = false;
			float fitness = 0;

		public:
			Agent(GenomeCPtr&& genome) : genome(std::move(genome)), isFinished(false), fitness(0.0f) {};
			~Agent() = default;
			Agent(const Agent&) = delete;
			Agent& operator=(const Agent&) = delete;
			Agent(const Agent&&) = delete;
			Agent& operator=(const Agent&&) = delete;

			virtual bool step() = 0;
			virtual void render(sf::RenderWindow* window) = 0;
			const GenomeCPtr& getGenome() const { return this->genome; };
			bool getFinished() const { return this->isFinished; };
			float getFitness() const { return this->fitness; };
		};

		class IGenepool
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

		using IGenepoolPtr = std::unique_ptr<IGenepool>;

		template<class TGenome, class TAgent> // TGenome: Genome<TGenome>, TAgent: Agent<TGenome>
		class Genepool : public IGenepool
		{
		protected:
			using GenomeCPtr = std::shared_ptr<const TGenome>;
			using AgentPtr = std::unique_ptr<TAgent>;

			bool enableMultithreadedStepEvaluation = false;
			bool enableMultithreadedFullEvaluation = false;
			bool syncMultithreadedSteps = false;

			bool isInitialized = false;
			int populationSize = 0;
			float mutationRate = 0.0f;

			ThreadPool threadPool;
			int generationNumber = 0;
			int generationStepNumber = 0;
			std::vector<AgentPtr> currentGeneration;
			bool isGenerationEvaluated = false;
			GenomeCPtr bestData = nullptr;
			float bestFitness = 0.0f;

			virtual GenomeCPtr createGenome() const { return std::make_shared<TGenome>(); }

			virtual AgentPtr createAgent(GenomeCPtr&& data) const { return std::make_unique<TAgent>(std::move(data)); }

		public:
			Genepool(bool enableMultithreadedStepEvaluation = false, bool enableMultithreadedFullEvaluation = true, bool syncMultithreadedSteps = false)
			{
				if (enableMultithreadedFullEvaluation && enableMultithreadedStepEvaluation)
					throw std::runtime_error("tbml::GenepoolSimulation: Cannot have both enableMultithreadedFullEvaluation and enableMultithreadedStepEvaluation.");
				if (syncMultithreadedSteps && !enableMultithreadedFullEvaluation)
					throw std::runtime_error("tbml::GenepoolSimulation: Cannot have syncMultithreadedSteps without enableMultithreadedFullEvaluation.");
				this->enableMultithreadedStepEvaluation = enableMultithreadedStepEvaluation;
				this->enableMultithreadedFullEvaluation = enableMultithreadedFullEvaluation;
				this->syncMultithreadedSteps = syncMultithreadedSteps;
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
					GenomeCPtr genome = createGenome();
					AgentPtr agent = createAgent(std::move(genome));
					this->currentGeneration.push_back(std::move(agent));
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
				auto evaluateSubset = [&](bool step, int start, int end)
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
							threadResults[i] = this->threadPool.enqueue([=] { return evaluateSubset(step || syncMultithreadedSteps, startIndex, endIndex); });
						}

						for (auto&& result : threadResults) allFinished &= result.get();
						this->generationStepNumber++;
					} while (!step && !allFinished);
				}

				// Process generation (single-threaded)
				else
				{
					allFinished = evaluateSubset(step, 0, this->currentGeneration.size());
					this->generationStepNumber++;
				}

				std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
				auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
				std::cout << "Processed Generation Step: " << us.count() / 1000.0f << "ms" << std::endl;

				this->generationStepNumber++;
				if (allFinished) this->isGenerationEvaluated = true;
			}

			void iterateGeneration()
			{
				if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot iterateGeneration because uninitialized.");
				if (!this->isGenerationEvaluated) return;

				// Sort generation and get best data
				std::sort(this->currentGeneration.begin(), this->currentGeneration.end(), [this](const auto& a, const auto& b) { return a->getFitness() > b->getFitness(); });
				const AgentPtr& bestInstance = this->currentGeneration[0];
				this->bestData = GenomeCPtr(bestInstance->getGenome());
				this->bestFitness = bestInstance->getFitness();

				std::cout << "Generation: " << this->generationNumber << ", best fitness: " << this->bestFitness << std::endl;

				// Initialize next generation with new instance of best data
				std::vector<AgentPtr> nextGeneration;
				nextGeneration.push_back(std::move(createAgent(std::move(GenomeCPtr(this->bestData)))));

				// Selection helper function to pick parents
				int selectAmount = static_cast<int>(ceil(this->currentGeneration.size() / 2.0f));
				auto transformFitness = [](float f) { return f * f; };
				float totalFitness = 0.0f;
				for (int i = 0; i < selectAmount; i++) totalFitness += transformFitness(this->currentGeneration[i]->getFitness());
				const auto& pickWeightedParent = [&]()
				{
					float r = fn::getRandomFloat() * totalFitness;
					float cumSum = 0.0f;
					for (int i = 0; i < selectAmount; i++)
					{
						cumSum += transformFitness(this->currentGeneration[i]->getFitness());
						if (r <= cumSum) return this->currentGeneration[i]->getGenome();
					}
					return this->currentGeneration[selectAmount - 1]->getGenome();
				};

				for (int i = 0; i < this->populationSize - 1; i++)
				{
					// [SELECTION] Pick 2 parents from previous generation
					const GenomeCPtr& parentDataA = pickWeightedParent();
					const GenomeCPtr& parentDataB = pickWeightedParent();

					// [CROSSOVER], [MUTATION] Crossover and mutate new child data
					GenomeCPtr childData = parentDataA->crossover(parentDataB, this->mutationRate);
					nextGeneration.push_back(std::move(createAgent(std::move(childData))));
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

		class GenepoolController
		{
		protected:
			IGenepoolPtr genepool = nullptr;

			bool isRunning = false;
			bool autoStepEvaluate = false;
			bool autoFullEvaluate = false;
			bool autoIterate = false;

		public:
			GenepoolController() {}

			GenepoolController(IGenepoolPtr&& genepool)
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

			const IGenepoolPtr& getGenepool() const { return this->genepool; }

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
}
