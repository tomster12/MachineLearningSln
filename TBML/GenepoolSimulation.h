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
		// TGenome: Self
		template<class TGenome>
		class Genome
		{
		public:
			using GenomeCPtr = std::shared_ptr<const TGenome>;

		protected:
			Genome() = default;
			~Genome() = default;

		public:
			Genome(const Genome&) = delete;
			Genome& operator=(const Genome&) = delete;
			Genome(const Genome&&) = delete;
			Genome& operator=(const Genome&&) = delete;
			virtual GenomeCPtr crossover(const GenomeCPtr& otherGenome, float mutateChance = 0.0f) const = 0;
		};

		// TGenome: Genome<TGenome>
		template<class TGenome>
		class Agent
		{
		public:
			using GenomeCPtr = std::shared_ptr<const TGenome>;

		protected:
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
			virtual void configThreading(bool enableMultithreadedStepEvaluation = false, bool enableMultithreadedFullEvaluation = false, bool syncMultithreadedSteps = false) = 0;
			virtual void resetGenepool(int populationSize, float mutationRate) = 0;
			virtual void render(sf::RenderWindow* window) = 0;
			virtual void initializeGeneration() = 0;
			virtual void stepGeneration(bool step = false) = 0;
			virtual void iterateGeneration() = 0;
			virtual int getGenerationNumber() const = 0;
			virtual float getBestFitness() const = 0;
			virtual bool getGenepoolInitialized() const = 0;
			virtual bool getGenerationEvaluated() const = 0;
		};

		using IGenepoolPtr = std::unique_ptr<IGenepool>;

		// TGenome: Genome<TGenome>, TAgent: Agent<TGenome>
		template<class TGenome, class TAgent>
		class Genepool : public IGenepool
		{
		public:
			using GenomeCPtr = std::shared_ptr<const TGenome>;
			using AgentPtr = std::unique_ptr<TAgent>;

		protected:
			std::function<GenomeCPtr(void)> createGenomeFn;
			std::function<AgentPtr(GenomeCPtr&&)> createAgentFn;
			bool useThreadedStep = false;
			bool useThreadedFullStep = false;
			bool syncThreadedFullSteps = false;
			int populationSize = 0;
			float mutationRate = 0.0f;

			bool isGenepoolInitialized = false;
			bool isGenerationEvaluated = false;
			int currentGeneration = 0;
			int currentStep = 0;
			GenomeCPtr bestData = nullptr;
			float bestFitness = 0.0f;
			ThreadPool stepThreadPool;
			std::vector<AgentPtr> currentAgents;

			virtual GenomeCPtr createGenome() const { return createGenomeFn(); }

			virtual AgentPtr createAgent(GenomeCPtr&& data) const { return createAgentFn(std::move(data)); }

		public:
			void setCreateGenomeFn(std::function<GenomeCPtr(void)> createGenomeFn) { this->createGenomeFn = createGenomeFn; }

			void setCreateAgentFn(std::function<AgentPtr(GenomeCPtr&&)> createAgentFn) { this->createAgentFn = createAgentFn; }

			void configThreading(bool enableMultithreadedStepEvaluation = false, bool enableMultithreadedFullEvaluation = false, bool syncMultithreadedSteps = false)
			{
				if (enableMultithreadedFullEvaluation && enableMultithreadedStepEvaluation)
					throw std::runtime_error("tbml::GenepoolSimulation: Cannot have both enableMultithreadedFullEvaluation and enableMultithreadedStepEvaluation.");
				if (syncMultithreadedSteps && !enableMultithreadedFullEvaluation)
					throw std::runtime_error("tbml::GenepoolSimulation: Cannot have syncMultithreadedSteps without enableMultithreadedFullEvaluation.");

				this->useThreadedStep = enableMultithreadedStepEvaluation;
				this->useThreadedFullStep = enableMultithreadedFullEvaluation;
				this->syncThreadedFullSteps = syncMultithreadedSteps;
			}

			void resetGenepool(int populationSize, float mutationRate)
			{
				// [INITIALIZATION] Initialize new instances
				this->currentAgents.clear();
				for (int i = 0; i < populationSize; i++)
				{
					GenomeCPtr genome = createGenome();
					AgentPtr agent = createAgent(std::move(genome));
					this->currentAgents.push_back(std::move(agent));
				}

				this->isGenepoolInitialized = true;
				this->populationSize = populationSize;
				this->mutationRate = mutationRate;

				this->currentGeneration = 1;
				this->currentStep = 0;
				this->isGenerationEvaluated = false;

				initializeGeneration();
			};

			void initializeGeneration() {}

			void stepGeneration(bool step)
			{
				if (!this->isGenepoolInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot evaluateGeneration because uninitialized.");
				if (this->isGenerationEvaluated) return;

				// Helper function captures generation
				auto evaluateSubset = [&](bool step, int start, int end)
				{
					bool allFinished;
					do
					{
						allFinished = true;
						for (int i = start; i < end; i++) allFinished &= this->currentAgents[i]->step();
					} while (!step && !allFinished);
					return allFinished;
				};

				std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
				bool allFinished = true;

				// Process generation (multi-threaded)
				if ((this->useThreadedStep && step) || (this->useThreadedFullStep && !step))
				{
					size_t threadCount = static_cast<size_t>(std::min(static_cast<int>(stepThreadPool.size()), this->populationSize));
					std::vector<std::future<bool>> threadResults(threadCount);
					int subsetSize = static_cast<int>(ceil((float)this->populationSize / threadCount));
					do
					{
						for (size_t i = 0; i < threadCount; i++)
						{
							int startIndex = i * subsetSize;
							int endIndex = static_cast<int>(std::min(startIndex + subsetSize, this->populationSize));
							threadResults[i] = this->stepThreadPool.enqueue([=] { return evaluateSubset(step || syncThreadedFullSteps, startIndex, endIndex); });
						}

						for (auto&& result : threadResults) allFinished &= result.get();
						this->currentStep++;
					} while (!step && !allFinished);
				}

				// Process generation (single-threaded)
				else
				{
					allFinished = evaluateSubset(step, 0, this->currentAgents.size());
					this->currentStep++;
				}

				std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
				auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
				std::cout << "Processed Generation Step: " << us.count() / 1000.0f << "ms" << std::endl;

				this->currentStep++;
				if (allFinished) this->isGenerationEvaluated = true;
			}

			void iterateGeneration()
			{
				if (!this->isGenepoolInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot iterateGeneration because uninitialized.");
				if (!this->isGenerationEvaluated) return;

				// Sort generation and get best data
				std::sort(this->currentAgents.begin(), this->currentAgents.end(), [this](const auto& a, const auto& b) { return a->getFitness() > b->getFitness(); });
				const AgentPtr& bestInstance = this->currentAgents[0];
				this->bestData = GenomeCPtr(bestInstance->getGenome());
				this->bestFitness = bestInstance->getFitness();
				std::cout << "Generation: " << this->currentGeneration << ", best fitness: " << this->bestFitness << std::endl;

				// Initialize next generation with best generation, setup selection method
				std::vector<AgentPtr> nextGeneration;
				nextGeneration.push_back(std::move(createAgent(std::move(GenomeCPtr(this->bestData)))));
				int selectAmount = static_cast<int>(ceil(this->currentAgents.size() / 2.0f));
				auto transformFitness = [](float f) { return f * f; };
				float totalFitness = 0.0f;
				for (int i = 0; i < selectAmount; i++) totalFitness += transformFitness(this->currentAgents[i]->getFitness());
				const auto& pickWeightedParent = [&]()
				{
					float r = fn::getRandomFloat() * totalFitness;
					float cumSum = 0.0f;
					for (int i = 0; i < selectAmount; i++)
					{
						cumSum += transformFitness(this->currentAgents[i]->getFitness());
						if (r <= cumSum) return this->currentAgents[i]->getGenome();
					}
					return this->currentAgents[selectAmount - 1]->getGenome();
				};

				for (int i = 0; i < this->populationSize - 1; i++)
				{
					// [SELECTION] Pick 2 parents from previous generation
					const GenomeCPtr& parentDataA = pickWeightedParent();
					const GenomeCPtr& parentDataB = pickWeightedParent();

					// [CROSSOVER], [MUTATION] Crossover and mutate new child data
					GenomeCPtr childData = parentDataA->crossover(parentDataB, this->mutationRate);
					nextGeneration.push_back(createAgent(std::move(childData)));
				}

				// Set to new generation and update variables
				this->currentAgents = std::move(nextGeneration);
				this->currentGeneration++;
				this->isGenerationEvaluated = false;
				initializeGeneration();
			};

			void render(sf::RenderWindow* window)
			{
				if (!this->isGenepoolInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot render because uninitialized.");

				for (const auto& inst : currentAgents) inst->render(window);
			};

			int getGenerationNumber() const { return this->currentGeneration; }

			GenomeCPtr getBestData() const { return this->bestData; }

			float getBestFitness() const { return this->bestFitness; }

			bool getGenepoolInitialized() const { return this->isGenepoolInitialized; }

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
				if (!this->genepool->getGenepoolInitialized()) throw std::runtime_error("tbml::GenepoolSimulationController: Cannot update because uninitialized.");
				if (this->genepool->getGenerationEvaluated() || !this->isRunning) return;

				this->evaluateGeneration(!this->autoFullEvaluate);
			};

			void render(sf::RenderWindow* window)
			{
				if (!this->genepool->getGenepoolInitialized()) throw std::runtime_error("tbml::GenepoolSimulation: Cannot render because uninitialized.");

				this->genepool->render(window);
			};

			void evaluateGeneration(bool step = false)
			{
				if (!this->genepool->getGenepoolInitialized()) throw std::runtime_error("tbml::GenepoolSimulationController: Cannot evaluateGeneration because uninitialized.");
				if (this->genepool->getGenerationEvaluated()) return;

				this->genepool->stepGeneration(step);
				if (this->autoIterate && this->genepool->getGenerationEvaluated()) this->iterateGeneration();
			}

			void iterateGeneration()
			{
				if (!this->genepool->getGenepoolInitialized()) throw std::runtime_error("tbml::GenepoolSimulationController: Cannot iterateGeneration because uninitialized.");
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
