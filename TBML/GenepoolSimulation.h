
#pragma once

#include<algorithm>
#include "UtilityFunctions.h"
#include "ThreadPool.h"


namespace tbml
{
	class IGenepoolSimulation
	{
	public:
		virtual void update() = 0;
		virtual void render(sf::RenderWindow* window) = 0;

		virtual void initGenepool(int generationCount, float mutationRate) = 0;
		virtual void initGeneration() = 0;
		virtual void startGeneration() = 0;
		virtual void pauseGeneration() = 0;
		virtual void fullStepGeneration() = 0;
		virtual void renderGeneration(sf::RenderWindow* window) = 0;
		virtual void singleStepGeneration() = 0;
		virtual void finishGeneration() = 0;

		virtual bool getAutoStart() = 0;
		virtual bool getAutoFinish() = 0;
		virtual bool getAutoComplete() = 0;

		virtual void setAutoStart(bool autoStart) = 0;
		virtual void setAutoFinish(bool autoFinish) = 0;
		virtual void setAutoComplete(bool autoComplete) = 0;
	};


	template<class D> // D: GeneticData<D>
	class GeneticData
	{
	public:
		virtual void randomize() = 0;
		virtual void mutate(float chance) = 0;
		virtual D* crossover(D* data) = 0;
	};


	template<class D> // D: GeneticData<D>
	class GeneticInstance
	{
	protected:
		D* geneticData;
		bool instanceFinished;
		float instanceFitness;

	public:
		GeneticInstance(D* geneticData) : geneticData(geneticData), instanceFinished(false), instanceFitness(0.0f) {};

		virtual bool step() = 0;
		virtual void render(sf::RenderWindow* window) = 0;

		virtual bool getInstanceFinished() = 0;
		virtual float getInstanceFitness() = 0;
	};


	template<class D, class I> // D: GeneticData<D>, I: GeneticInstance<D>
	class GenepoolSimulation : public IGenepoolSimulation
	{
	protected:
		struct GeneticEvaluation
		{
			D* data;
			I* instance;
			bool isFinished;
			float fitness;
		};


		int generationCount;
		float mutationRate;
		bool isInitialized;
		int generationNumber;
		int generationStep;
		bool isRunning;
		bool isFinished;
		bool autoStart;
		bool autoFinish;
		bool autoComplete;
		std::vector<GeneticEvaluation> currentGeneration;
		ThreadPool threadPool;


		virtual D* createData()
		{
			// Create, randomize and return data
			D* data = new D();
			data->randomize();
			return data;
		}

		virtual I* createInstance(D* data)
		{
			// Create and return instance
			I* inst = new I(data);
			return inst;
		}


	public:
		GenepoolSimulation()
		{
			// Initialize variables
			this->generationCount = 0;
			this->mutationRate = 0;
			this->isInitialized = false;
			this->generationNumber = 0;
			this->generationStep = 0;
			this->isRunning = false;
			this->isFinished = false;
			this->autoStart = false;
			this->autoFinish = false;
			this->autoComplete = false;
		}

		~GenepoolSimulation()
		{
			// Delete all data / instances
			if (this->isInitialized)
			{
				for (int i = 0; i < this->generationCount; i++)
				{
					delete this->currentGeneration[i].data;
					delete this->currentGeneration[i].instance;
				}
			}
		}


		void update()
		{
			// Step / complete generation then check for auto finish
			if (!this->isInitialized) return;
			if (this->isFinished || !this->isRunning) return;
			if (this->autoComplete) fullStepGeneration();
			else singleStepGeneration();
			if (this->autoFinish && this->isFinished) finishGeneration();
		};

		void render(sf::RenderWindow* window)
		{
			// Render all current generation
			if (!this->isInitialized) return;
			this->renderGeneration(window);
		};


		void initGenepool(int generationCount, float mutationRate)
		{
			// Delete all data / instances
			if (this->isInitialized)
			{
				for (int i = 0; i < this->generationCount; i++)
				{
					delete this->currentGeneration[i].data;
					delete this->currentGeneration[i].instance;
				}
			}

			// [INITIALIZATION] Initialize all data and instances
			for (int i = 0; i < generationCount; i++)
			{
				D* data = createData();
				I* inst = createInstance(data);
				this->currentGeneration.push_back({ data, inst, false, 0.0f });
			}

			// Initialize variables
			this->generationCount = generationCount;
			this->mutationRate = mutationRate;
			this->isInitialized = true;
			this->generationNumber = 1;
			this->generationStep = 0;
			this->isRunning = false;
			this->isFinished = false;
			this->autoStart = false;
			this->autoFinish = false;
			this->autoComplete = false;
			initGeneration();
		};

		void initGeneration() {}

		void startGeneration()
		{
			// Start current generation
			if (!this->isInitialized) return;
			if (this->isRunning || this->isFinished) return;
			this->isRunning = true;
		};

		void pauseGeneration()
		{
			// Pause current generation
			if (!this->isInitialized) return;
			if (!this->isRunning || this->isFinished) return;
			this->isRunning = false;
		};

		void fullStepGeneration()
		{
			if (!this->isInitialized) return;

			// Only run updates if not finished
			if (!this->isFinished)
			{
				// Process with multiple threads
				auto process = [=](std::vector<GeneticEvaluation&> generationSubset)
				{
					bool allFinished = false;
					while (!allFinished)
					{
						allFinished = true;
						for (GeneticEvaluation& eval : generationSubset)
						{
							if (eval.instance->step()) eval.fitness = eval.instance->getInstanceFitness();
							else allFinished = false;
						}
					}
				};

				// Set of all the threads
				size_t threadCount = threadPool.size();
				const int gap = generationCount / threadCount;
				if (generationCount % threadCount != 0) threadCount++;
				std::vector<std::future<void>> results(threadCount);
				for (size_t i = 0; i < threadCount; i++)
				{
					int start = i * gap;
					int end = static_cast<int>(std::min(start + gap, generationCount));

					std::vector<GeneticEvaluation&> generationSubset(end - gap);
					for (int i = start; i < end; i++) { generationSubset[i] = &currentGeneration[i]; }

					results[i] = threadPool.enqueue([=] { process(generationSubset); });
				}

				// Wait for all threads to finish
				for (auto&& result : results) result.get();
				
				// Finish up running
				this->isFinished = true;
				this->isRunning = false;
			}
			
			// Auto finish if full stepping
			if (this->autoFinish) finishGeneration();
		}

		void renderGeneration(sf::RenderWindow* window)
		{
			// Render all instances
			for (auto& eval : currentGeneration) eval.instance->render(window);
		}

		void singleStepGeneration()
		{
			if (!this->isInitialized) return;
			if (this->isFinished) return;

			// Step all unfinished instances
			bool allFinished = true;
			for (GeneticEvaluation& eval : currentGeneration)
			{
				if (eval.instance->step()) eval.fitness = eval.instance->getInstanceFitness();
				else allFinished = false;
			}

			// Once all finished update variables
			this->generationStep++;
			if (allFinished)
			{
				this->isFinished = true;
				this->isRunning = false;
			}
		}

		void finishGeneration()
		{
			if (!this->isInitialized) return;
			if (this->isRunning || !this->isFinished) return;

			// [SELECTION] Sort, then cull the bottom half of the generation
			std::sort(this->currentGeneration.begin(), this->currentGeneration.end(), [this](auto a, auto b) { return a.fitness < b.fitness; });
			int amount = static_cast<int>(this->currentGeneration.size() / 2);
			std::vector<GeneticEvaluation> selectedGeneration(this->currentGeneration.begin() + amount, this->currentGeneration.end());
			std::cout << "Generation: " << this->generationNumber << ", best fitness: " << selectedGeneration[selectedGeneration.size() - 1].fitness << std::endl;
			kjhojdf

			// setup parent selection function
			float totalFitness = 0.0f;
			for (auto& eval : selectedGeneration) totalFitness += eval.fitness;
			auto pickParent = [selectedGeneration, totalFitness](float r)
			{
				float cumSum = 0.0f;
				for (auto& eval : selectedGeneration)
				{
					cumSum += eval.fitness / totalFitness;
					if (r <= cumSum) return eval.data;
				}
				return selectedGeneration[selectedGeneration.size() - 1].data;
			};

			// Create the next generation
			std::vector<GeneticEvaluation> newGeneration;
			for (int i = 0; i < this->generationCount; i++)
			{
				// [CROSSOVER], [MUTATION] Create new genetic data
				D* dataA = pickParent(getRandomFloat());
				D* dataB = pickParent(getRandomFloat());
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
			initGeneration();

			// Handle auto start
			if (this->autoStart) startGeneration();
		};


		bool getAutoStart() { return this->autoStart; };

		bool getAutoFinish() { return this->autoFinish; };

		bool getAutoComplete() { return this->autoComplete; };


		void setAutoStart(bool autoStart) { this->autoStart = autoStart; if (!this->isRunning && this->autoStart) this->startGeneration(); };

		void setAutoFinish(bool autoFinish) { this->autoFinish = autoFinish; if (this->isFinished && this->autoFinish) this->finishGeneration(); }

		void setAutoComplete(bool autoComplete) { this->autoComplete = autoComplete; if (this->autoComplete) this->fullStepGeneration(); }
	};
}
