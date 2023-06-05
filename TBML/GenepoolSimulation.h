
#pragma once

#include "UtilityFunctions.h"
#include "ThreadPool.h"


namespace tbml
{
	class IGenepoolSimulation
	{
	public:
		virtual void update() = 0;
		virtual void render(sf::RenderWindow* window) = 0;

		virtual void restartGenepool(int generationCount, float mutationRate) = 0;
		virtual void initGeneration() = 0;
		virtual void stepGeneration() = 0;
		virtual void processGeneration() = 0;
		virtual void finishGeneration() = 0;

		virtual bool getAutoStep() = 0;
		virtual bool getAutoFinish() = 0;
		virtual bool getAutoProcess() = 0;

		virtual void setStepping(bool isStepping) = 0;
		virtual void setAutoStep(bool autoStep) = 0;
		virtual void setAutoFinish(bool autoFinish) = 0;
		virtual void setAutoProcess(bool autoProcess) = 0;
	};


	template<class D> // D: GeneticData<D>
	class GeneticData
	{
	public:
		virtual void randomize() = 0;
		virtual void mutate(float chance) = 0;
		virtual D* crossover(D* data) = 0;
		virtual D* clone() = 0;
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
		~GeneticInstance() { delete this->geneticData; }

		virtual bool step() = 0;
		virtual void render(sf::RenderWindow* window) = 0;

		D* getData() { return this->geneticData; };
		bool getFinished() { return this->instanceFinished; };
		float getFitness() { return this->instanceFitness; };
	};


	template<class D, class I> // D: GeneticData<D>, I: GeneticInstance<D>
	class GenepoolSimulation : public IGenepoolSimulation
	{
	protected:
		bool linkedSteps;
		bool enableMultithreadedStep;
		bool enableMultithreadedProcess;
		int generationCount;
		float mutationRate;
		bool isInitialized;
		int generationNumber;
		int generationStep;
		bool isStepping;
		bool isFinished;
		bool autoStep;
		bool autoFinish;
		bool autoProcess;

		ThreadPool threadPool;
		std::vector<I*> currentGeneration;
		D* bestCurrentData;
		float bestCurrentFitness;


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
		GenepoolSimulation(bool linkedSteps = false, bool enableMultithreadedStep = false, bool enableMultithreadedProcess = true)
		{
			// Initialize variables
			this->linkedSteps = linkedSteps;
			this->enableMultithreadedStep = enableMultithreadedStep;
			this->enableMultithreadedProcess = enableMultithreadedProcess;
			this->generationCount = 0;
			this->mutationRate = 0;
			this->isInitialized = false;
			this->generationNumber = 0;
			this->generationStep = 0;
			this->isStepping = false;
			this->isFinished = false;
			this->autoStep = false;
			this->autoFinish = false;
			this->autoProcess = false;
		}

		~GenepoolSimulation()
		{
			// Delete all instances
			if (this->isInitialized)
			{
				for (int i = 0; i < generationCount; i++) delete this->currentGeneration[i];
			}
		}


		void update()
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot update because uninitialized.");

			// Step / complete generation then check for auto finish
			if (this->isFinished || !this->isStepping) return;
			if (this->autoProcess) processGeneration();
			else stepGeneration();
			if (this->autoFinish && this->isFinished) finishGeneration();
		};

		void render(sf::RenderWindow* window)
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot render because uninitialized.");

			// Render all instances
			for (auto inst : currentGeneration) inst->render(window);
		};


		void restartGenepool(int generationCount, float mutationRate)
		{
			// Delete all current instances
			if (this->isInitialized)
			{
				for (int i = 0; i < generationCount; i++) delete this->currentGeneration[i];
			}

			// [INITIALIZATION] Initialize new instances
			this->currentGeneration.clear();
			for (int i = 0; i < generationCount; i++)
			{
				D* data = createData();
				this->currentGeneration.push_back(createInstance(data));
			}

			// Initialize variables
			this->generationCount = generationCount;
			this->mutationRate = mutationRate;
			this->isInitialized = true;
			this->generationNumber = 1;
			this->generationStep = 0;
			this->isStepping = false;
			this->isFinished = false;
			this->autoStep = false;
			this->autoFinish = false;
			this->autoProcess = false;
			initGeneration();
		};

		void initGeneration() {}

		void stepGeneration()
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot stepGeneration because uninitialized.");

			// Only step if currently stepping and not finished
			if (!this->isStepping || this->isFinished) return;

			// Helper function
			auto stepSubset = [=](std::vector<I*> subset)
			{
				bool allFinished = true;
				for (auto& inst : subset) allFinished &= inst->step();
				return allFinished;
			};

			// Split up generation and step each
			bool allFinished = true;
			if (this->enableMultithreadedStep)
			{
				size_t threadCount = static_cast<size_t>(std::min(static_cast<int>(threadPool.size()), this->generationCount));
				std::vector<std::future<bool>> results(threadCount);
				int gap = static_cast<int>(ceil((float)this->generationCount / threadCount));
				for (size_t i = 0; i < threadCount; i++)
				{
					int start = i * gap;
					int end = static_cast<int>(std::min(start + gap, this->generationCount));
					std::vector<I*> subset(this->currentGeneration.begin() + start, this->currentGeneration.begin() + end);
					results[i] = threadPool.enqueue([=] { return stepSubset(subset); });
				}
				for (auto&& result : results) allFinished &= result.get();
			}

			// Step entire generation without multithreading
			else allFinished = stepSubset(this->currentGeneration);

			// Once all finished update variables
			this->generationStep++;
			if (allFinished)
			{
				this->isFinished = true;
				this->isStepping = false;
			}
		}

		void processGeneration()
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot processGeneration because uninitialized.");

			// Only process if not finished
			if (this->isFinished) return;

			// Helper functions
			auto stepSubset = [=](std::vector<I*> subset)
			{
				bool allFinished = true;
				for (auto& inst : subset) allFinished &= inst->step();
				return allFinished;
			};

			auto processSubset = [=](std::vector<I*> subset)
			{
				for (bool allFinished = false; !allFinished;)
				{
					allFinished = true;
					for (auto& inst : subset) allFinished &= inst->step();
				}
				return true;
			};

			// Start timer
			//std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

			// Split up generation and process each
			if (this->enableMultithreadedProcess)
			{
				size_t threadCount = static_cast<size_t>(std::min(static_cast<int>(threadPool.size()), this->generationCount));
				std::vector<std::future<bool>> results(threadCount);
				int gap = static_cast<int>(ceil((float)this->generationCount / threadCount));
				for (bool allFinished = false; !allFinished;)
				{
					for (size_t i = 0; i < threadCount; i++)
					{
						int start = i * gap;
						int end = static_cast<int>(std::min(start + gap, this->generationCount));
						std::vector<I*> subset(this->currentGeneration.begin() + start, this->currentGeneration.begin() + end);
						if (this->linkedSteps) results[i] = threadPool.enqueue([=] { return stepSubset(subset); });
						else results[i] = threadPool.enqueue([=] { return processSubset(subset); });
					}
					allFinished = true;
					for (auto&& result : results) allFinished &= result.get();
					if (this->linkedSteps) this->generationStep++;
				}
			}

			// Process full generation without multithreading
			else processSubset(this->currentGeneration);

			// Stop timer and print
			//std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
			//auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
			//std::cout << "Processed generation: " << us.count() / 1000.0f << "ms" << std::endl;

			// Finish up running
			this->isFinished = true;
			this->isStepping = false;
			if (this->autoFinish) finishGeneration();
		}

		void finishGeneration()
		{
			if (!this->isInitialized) throw std::runtime_error("tbml::GenepoolSimulation: Cannot finishGeneration because uninitialized.");

			// Only finish if not currently stepping and finished
			if (this->isStepping || !this->isFinished) return;

			// [SELECTION] Sort, then cull the bottom half of the generation
			std::sort(this->currentGeneration.begin(), this->currentGeneration.end(), [this](auto a, auto b) { return a->getFitness() < b->getFitness(); });
			int selectAmount = static_cast<int>(ceil(this->currentGeneration.size() / 2.0f));
			std::vector<I*> selectedGeneration(this->currentGeneration.end() - selectAmount, this->currentGeneration.end());

			// get best instance
			I* bestInstance = selectedGeneration[selectAmount - 1];
			std::cout << "Generation: " << this->generationNumber << ", best fitness: " << bestInstance->getFitness() << std::endl;

			// setup parent selection function
			auto transformFitness = [](float f) { return f * f; };
			float totalFitness = 0.0f;
			for (auto& inst : selectedGeneration) totalFitness += (transformFitness(inst->getFitness()));
			auto pickRandomData = [selectedGeneration, totalFitness, transformFitness]()
			{
				float r = getRandomFloat() * totalFitness;
				float cumSum = 0.0f;
				for (auto& inst : selectedGeneration)
				{
					cumSum += transformFitness(inst->getFitness());
					if (r <= cumSum) return inst->getData();
				}
				return selectedGeneration[selectedGeneration.size() - 1]->getData();
			};

			// Create the next generation
			std::vector<I*> newGeneration(this->generationCount);
			for (int i = 0; i < this->generationCount - 1; i++)
			{
				// [CROSSOVER], [MUTATION] Create new genetic data
				D* dataA = pickRandomData();
				D* dataB = pickRandomData();
				D* newData = dataA->crossover(dataB);
				newData->mutate(this->mutationRate);
				newGeneration[i] = createInstance(newData);
			}

			// Keep the best data and delete the old instances
			newGeneration[this->generationCount - 1] = createInstance(bestInstance->getData()->clone());
			for (int i = 0; i < generationCount; i++) delete this->currentGeneration[i];

			// Set to new generation and update variables
			this->currentGeneration = newGeneration;
			this->generationNumber++;
			this->isFinished = false;
			initGeneration();

			// Handle auto start
			if (this->autoStep) setStepping(true);
		};


		bool getAutoStep() { return this->autoStep; };

		bool getAutoFinish() { return this->autoFinish; };

		bool getAutoProcess() { return this->autoProcess; };


		void setStepping(bool isStepping) { if (!this->isFinished) this->isStepping = isStepping; }

		void setAutoStep(bool autoStep) { this->autoStep = autoStep; if (!this->isStepping && this->autoStep) this->setStepping(true); };

		void setAutoProcess(bool autoProcess) { this->autoProcess = autoProcess; if (this->autoProcess) this->processGeneration(); }

		void setAutoFinish(bool autoFinish) { this->autoFinish = autoFinish; if (this->isFinished && this->autoFinish) this->finishGeneration(); }
	};
}
