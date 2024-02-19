#pragma once

#include <mutex>
#include "Matrix.h"
#include "NeuralNetwork.h"

namespace tbml
{
	namespace nn
	{
		struct BackpropogateCache
		{
			std::vector<Matrix> pdToNeuronIn;
			std::vector<Matrix> pdToNeuronOut;
			std::vector<std::vector<Matrix>> pdToWeights;
			std::vector<std::vector<Matrix>> pdToBias;
		};

		struct TrainingConfig { int epochs = 20; int batchSize = -1; float learningRate = 0.1f; float momentumRate = 0.1f; float errorExit = 0.0f; int logLevel = 0; };

		class SupervisedNetwork : public NeuralNetwork
		{
		public:
			SupervisedNetwork(std::vector<size_t> layerSizes, WeightInitType weightInitType = RANDOM);
			SupervisedNetwork(std::vector<size_t> layerSizes, std::vector<fn::ActivationFunction> actFns, WeightInitType weightInitType = RANDOM);
			SupervisedNetwork(std::vector<size_t> layerSizes, fn::LossFunction lossFn, WeightInitType weightInitType = RANDOM);
			SupervisedNetwork(std::vector<size_t> layerSizes, std::vector<fn::ActivationFunction> actFns, fn::LossFunction lossFn, WeightInitType weightInitType = RANDOM);

			void train(const Matrix& input, const Matrix& expected, const TrainingConfig& config);

		private:
			static const int MAX_MAX_ITERATIONS = 1'000'000;
			fn::LossFunction lossFn;

			float trainBatch(const Matrix& input, const Matrix& expected, const TrainingConfig& config, std::vector<Matrix>& pdWeightsMomentum, std::vector<Matrix>& pdBiasMomentum);
			void backpropogate(const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const;
			Matrix const& calculatePdErrorToIn(size_t layer, const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const;
			Matrix const& calculatePdErrorToOut(size_t layer, const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const;
			BackpropogateCache preinitializeBackpropagationCache(int inputCount) const;
		};
	}
}
