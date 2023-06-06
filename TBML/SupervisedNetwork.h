
#pragma once

#include<mutex>
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
		SupervisedNetwork(std::vector<size_t> layerSizes_);
		SupervisedNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float));
		SupervisedNetwork(std::vector<size_t> layerSizes_, float (*calcError_)(const Matrix&, const Matrix&), Matrix(*calcErrorPd_)(const Matrix&, const Matrix&));
		SupervisedNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float), float (*calcError_)(const Matrix&, const Matrix&), Matrix(*calcErrorPd_)(const Matrix&, const Matrix&));

		void train(const Matrix& input, const Matrix& expected, const TrainingConfig& config);

	private:
		static const int MAX_MAX_ITERATIONS = 1'000'000;
		float (*activatorPd)(float);
		float (*calcError)(const Matrix&, const Matrix&);
		Matrix(*calcErrorPd)(const Matrix&, const Matrix&);

		float trainBatch(const Matrix& input, const Matrix& expected, const TrainingConfig& config, std::vector<Matrix>& pdWeightsMomentum, std::vector<Matrix>& pdBiasMomentum, std::mutex& updateMutex);
		void backpropogate(const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropgateCache) const;
		void calculatePdErrorToIn(size_t layer, const PropogateCache& predictedCache, BackpropogateCache& backpropgateCache) const;
		void calculatePdErrorToOut(size_t layer, const PropogateCache& predictedCache, BackpropogateCache& backpropgateCache) const;
	};
}
