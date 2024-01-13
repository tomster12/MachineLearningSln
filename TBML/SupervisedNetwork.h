#pragma once

#include <mutex>
#include "Matrix.h"
#include "NeuralNetwork.h"

namespace tbml
{
	struct BackpropogateCache
	{
		Matrix pdOut;
		std::vector<Matrix> pdNeuronIn;
		std::vector<Matrix> pdNeuronOut;
		std::vector<std::vector<Matrix>> pdWeights;
		std::vector<std::vector<Matrix>> pdBias;
	};

	struct TrainingConfig { int epochs = 20; int batchSize = -1; float learningRate = 0.1f; float momentumRate = 0.1f; float errorExit = 0.0f; int logLevel = 0; };

	class SupervisedNetwork : public NeuralNetwork
	{
	public:
		SupervisedNetwork(std::vector<size_t> layerSizes);
		SupervisedNetwork(std::vector<size_t> layerSizes, std::vector<fns::ActivationFunction> actFns);
		SupervisedNetwork(std::vector<size_t> layerSizes, fns::ErrorFunction errorFn);
		SupervisedNetwork(std::vector<size_t> layerSizes, std::vector<fns::ActivationFunction> actFns, fns::ErrorFunction errorFn);

		void train(const Matrix& input, const Matrix& expected, const TrainingConfig& config);

	private:
		static const int MAX_MAX_ITERATIONS = 1'000'000;
		fns::ErrorFunction errorFn;

		float trainBatch(const Matrix& input, const Matrix& expected, const TrainingConfig& config, std::vector<Matrix>& pdWeightsMomentum, std::vector<Matrix>& pdBiasMomentum, std::mutex& updateMutex);
		void backpropogate(const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const;
		void calculatePdErrorToIn(size_t layer, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const;
		void calculatePdErrorToOut(size_t layer, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const;
	};
}
