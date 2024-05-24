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

			Genome(const Genome&) = delete;
			Genome& operator=(const Genome&) = delete;
			Genome(const Genome&&) = delete;
			Genome& operator=(const Genome&&) = delete;
			virtual GenomeCPtr crossover(const GenomeCPtr& otherGenome, float mutateChance = 0.0f) const = 0;

		protected:
			Genome() = default;
			~Genome() = default;
		};

		// TGenome: Genome<TGenome>
		template<class TGenome>
		class Agent
		{
		public:
			using GenomeCPtr = std::shared_ptr<const TGenome>;

			Agent(GenomeCPtr&& genome) : genome(std::move(genome)), isFinished(false), fitness(0.0f) {};
			~Agent() = default;
			Agent(const Agent&) = delete;
			Agent& operator=(const Agent&) = delete;
			Agent(const Agent&&) = delete;
			Agent& operator=(const Agent&&) = delete;
			virtual bool evaluate() = 0;
			virtual void render(sf::RenderWindow* window) = 0;
			const GenomeCPtr& getGenome() const { return this->genome; };
			bool getFinished() const { return this->isFinished; };
			float getFitness() const { return this->fitness; };

		protected:
			const GenomeCPtr genome;
			bool isFinished = false;
			float fitness = 0;
		};

		class IGenepool
		{
		public:
			virtual void configThreading(bool enableMultithreadedStepEvaluation = false, bool enableMultithreadedFullEvaluation = false, bool syncMultithreadedSteps = false) = 0;
			virtual void resetGenepool(int populationSize, float mutationRate) = 0;
			virtual void render(sf::RenderWindow* window) = 0;
			virtual void initializeGeneration() = 0;
			virtual void evaluateGeneration(bool step = false) = 0;
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
				this->agentPopulation.clear();
				for (int i = 0; i < populationSize; i++)
				{
					GenomeCPtr genome = createGenome();
					AgentPtr agent = createAgent(std::move(genome));
					this->agentPopulation.push_back(std::move(agent));
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

			void evaluateGeneration(bool singleStep)
			{
				if (!this->isGenepoolInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot evaluateGeneration because uninitialized.");
				if (this->isGenerationEvaluated) return;

				// Process generation (multi-threaded)
				if ((this->useThreadedStep && singleStep) || (this->useThreadedFullStep && !singleStep))
				{
					// Helper function to evaluate a subset (captures generation)
					auto evaluateSubset = [&](bool singleStep, int start, int end)
					{
						bool subsetEvaluated = false;
						while (!subsetEvaluated)
						{
							subsetEvaluated = true;
							for (int i = start; i < end; i++) subsetEvaluated &= this->agentPopulation[i]->evaluate();
							if (singleStep) break;
						}
						return subsetEvaluated;
					};

					bool allFinished = true;
					size_t threadCount = static_cast<size_t>(std::min(static_cast<int>(evaluateThreadPool.size()), this->populationSize));
					std::vector<std::future<bool>> threadResults(threadCount);
					int subsetSize = static_cast<int>(ceil((float)this->populationSize / threadCount));

					while (!this->isGenerationEvaluated)
					{
						for (size_t i = 0; i < threadCount; i++)
						{
							int startIndex = i * subsetSize;
							int endIndex = static_cast<int>(std::min(startIndex + subsetSize, this->populationSize));
							threadResults[i] = this->evaluateThreadPool.enqueue([=] { return evaluateSubset(singleStep || syncThreadedFullSteps, startIndex, endIndex); });
						}

						this->isGenerationEvaluated = true;
						for (auto&& result : threadResults) this->isGenerationEvaluated &= result.get();
						this->currentStep++;
						if (singleStep) break;
					}
				}

				// Process generation (single-threaded)
				else
				{
					while (!this->isGenerationEvaluated)
					{
						this->isGenerationEvaluated = true;
						for (auto& inst : this->agentPopulation) this->isGenerationEvaluated &= inst->evaluate();
						this->currentStep++;
						if (singleStep) break;
					}
				}
			}

			void iterateGeneration()
			{
				if (!this->isGenepoolInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot iterateGeneration because uninitialized.");
				if (!this->isGenerationEvaluated) return;

				// Sort generation and extract best agent
				std::sort(this->agentPopulation.begin(), this->agentPopulation.end(), [this](const auto& a, const auto& b) { return a->getFitness() > b->getFitness(); });
				const AgentPtr& bestInstance = this->agentPopulation[0];
				this->bestData = GenomeCPtr(bestInstance->getGenome());
				this->bestFitness = bestInstance->getFitness();
				std::cout << "Generation: " << this->currentGeneration << ", best fitness: " << this->bestFitness << std::endl;

				// Initialize next generation with previous best
				std::vector<AgentPtr> nextGeneration;
				nextGeneration.push_back(std::move(createAgent(std::move(GenomeCPtr(this->bestData)))));

				// Setup selection for fitness proportional roulette selection
				int selectAmount = static_cast<int>(ceil(this->agentPopulation.size() / 2.0f));
				auto transformFitness = [](float f) { return f * f; };
				float totalFitness = 0.0f;
				for (int i = 0; i < selectAmount; i++) totalFitness += transformFitness(this->agentPopulation[i]->getFitness());
				const auto& pickWeightedParent = [&]()
				{
					float r = fn::getRandomFloat() * totalFitness;
					float cumSum = 0.0f;
					for (int i = 0; i < selectAmount; i++)
					{
						cumSum += transformFitness(this->agentPopulation[i]->getFitness());
						if (r <= cumSum) return this->agentPopulation[i]->getGenome();
					}
					return this->agentPopulation[selectAmount - 1]->getGenome();
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
				this->agentPopulation = std::move(nextGeneration);
				this->currentGeneration++;
				this->isGenerationEvaluated = false;
				initializeGeneration();
			};

			void render(sf::RenderWindow* window)
			{
				if (!this->isGenepoolInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot render because uninitialized.");
				for (const auto& inst : agentPopulation) inst->render(window);
			};

			int getGenerationNumber() const { return this->currentGeneration; }

			GenomeCPtr getBestData() const { return this->bestData; }

			float getBestFitness() const { return this->bestFitness; }

			bool getGenepoolInitialized() const { return this->isGenepoolInitialized; }

			bool getGenerationEvaluated() const { return this->isGenerationEvaluated; }

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
			ThreadPool evaluateThreadPool;
			std::vector<AgentPtr> agentPopulation;

			virtual GenomeCPtr createGenome() const { return createGenomeFn(); }

			virtual AgentPtr createAgent(GenomeCPtr&& data) const { return createAgentFn(std::move(data)); }
		};

		class GenepoolController
		{
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

				this->genepool->evaluateGeneration(step);
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

		protected:
			IGenepoolPtr genepool = nullptr;
			bool isRunning = false;
			bool autoStepEvaluate = false;
			bool autoFullEvaluate = false;
			bool autoIterate = false;
		};
	}
}
