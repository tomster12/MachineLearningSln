#include <iomanip>
#include "stdafx.h"
#include "SupervisedNetwork.h"
#include "Utility.h"
#include "Matrix.h"
#include "ThreadPool.h"

namespace tbml
{
	namespace nn
	{
		SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes, WeightInitType weightInitType)
			: SupervisedNetwork(layerSizes, fn::SquareError(), weightInitType)
		{}

		SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes, std::vector<fn::ActivationFunction> actFns, WeightInitType weightInitType)
			: SupervisedNetwork(layerSizes, actFns, fn::SquareError(), weightInitType)
		{}

		SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes, fn::ErrorFunction errorFn, WeightInitType weightInitType)
			: NeuralNetwork(layerSizes, weightInitType), errorFn(errorFn)
		{}

		SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes, std::vector<fn::ActivationFunction> actFns, fn::ErrorFunction errorFn, WeightInitType weightInitType)
			: NeuralNetwork(layerSizes, actFns, weightInitType), errorFn(errorFn)
		{}

		void SupervisedNetwork::train(const Matrix& input, const Matrix& expected, const TrainingConfig& config)
		{
			size_t batchCount;
			std::vector<Matrix> inputBatches, expectedBatches;

			// Do not batch data
			if (config.batchSize == -1)
			{
				inputBatches = std::vector<Matrix>({ input });
				expectedBatches = std::vector<Matrix>({ expected });
				batchCount = 1;
			}

			// Split input and expected into batches
			else
			{
				inputBatches = input.groupRows(config.batchSize);
				expectedBatches = expected.groupRows(config.batchSize);
				batchCount = config.batchSize;
			}

			// Initialize training loop variables
			std::vector<Matrix> weightsMomentum = std::vector<Matrix>(layerCount);
			std::vector<Matrix> biasMomentum = std::vector<Matrix>(layerCount);
			int maxEpochs = (config.epochs == -1 && config.errorExit > 0.0f) ? MAX_MAX_ITERATIONS : config.epochs;
			int epoch = 0;

			if (config.logLevel >= 1)
			{
				std::cout << "Training Started" << std::endl;
			}

			std::chrono::steady_clock::time_point tTrainStart = std::chrono::steady_clock::now();
			std::chrono::steady_clock::time_point tEpochStart = tTrainStart;
			std::chrono::steady_clock::time_point tBatchStart = tTrainStart;

			// For each epoch, for each mini-batch, train and then track error
			for (; epoch < maxEpochs; epoch++)
			{
				float epochLoss = 0.0f;
				for (size_t batch = 0; batch < batchCount; batch++)
				{
					float batchLoss = trainBatch(inputBatches[batch], expectedBatches[batch], config, weightsMomentum, biasMomentum);
					epochLoss += batchLoss / batchCount;

					if (config.logLevel == 3)
					{
						std::chrono::steady_clock::time_point tBatchFinish = std::chrono::steady_clock::now();

						Matrix predicted = propogate(inputBatches[batch]);
						float accuracy = fn::calculateAccuracy(predicted, expectedBatches[batch]);

						auto us = std::chrono::duration_cast<std::chrono::microseconds>(tBatchFinish - tBatchStart);
						std::cout << std::setw(20) << ("Batch = " + std::to_string(batch + 1) + " / " + std::to_string(batchCount))
							<< std::setw(24) << ("Epoch = " + std::to_string(epoch + 1) + " / " + std::to_string(maxEpochs))
							<< std::setw(24) << ("Duration = " + std::to_string(us.count() / 1000) + "ms")
							<< std::setw(29) << ("Batch Loss = " + std::to_string(batchLoss))
							<< std::setw(35) << ("Batch Accuracy = " + std::to_string(accuracy * 100) + "%")
							<< std::endl;

						tBatchStart = std::chrono::steady_clock::now();
					}
				}

				if (config.logLevel == 2)
				{
					std::chrono::steady_clock::time_point tEpochFinish = std::chrono::steady_clock::now();

					Matrix predicted = propogate(input);
					float accuracy = fn::calculateAccuracy(predicted, expected);

					auto us = std::chrono::duration_cast<std::chrono::microseconds>(tEpochFinish - tEpochStart);
					std::cout << std::setw(20) << ("Epoch = " + std::to_string(epoch + 1) + " / " + std::to_string(maxEpochs))
						<< std::setw(24) << ("Duration = " + std::to_string(us.count() / 1000) + "ms")
						<< std::setw(35) << ("Avg. Batch Loss = " + std::to_string(epochLoss))
						<< std::setw(35) << ("Epoch Accuracy = " + std::to_string(accuracy * 100) + "%")
						<< std::endl;

					tEpochStart = std::chrono::steady_clock::now();
				}

				// Exit early if error below certain amount
				if (epochLoss < config.errorExit) { epoch++; break; }
			}

			if (config.logLevel >= 1)
			{
				std::chrono::steady_clock::time_point tTrainFinish = std::chrono::steady_clock::now();

				Matrix predicted = propogate(input);
				float accuracy = fn::calculateAccuracy(predicted, expected);

				auto us = std::chrono::duration_cast<std::chrono::microseconds>(tTrainFinish - tTrainStart);
				std::cout << "Training Finished"
					<< std::setw(20) << ("Epochs = " + std::to_string(epoch))
					<< std::setw(26) << ("Duration = " + std::to_string(us.count() / 1000) + "ms")
					<< std::setw(35) << ("Train Accuracy = " + std::to_string(accuracy * 100) + "%")
					<< std::endl << std::endl;
			}
		}

		float SupervisedNetwork::trainBatch(const Matrix& input, const Matrix& expected, const TrainingConfig& config, std::vector<Matrix>& weightsMomentum, std::vector<Matrix>& biasMomentum)
		{
			// Forward propogation current batch
			PropogateCache predictedCache;
			propogate(input, predictedCache);

			// Backwards propogate using mini-batch gradient descent
			BackpropogateCache backpropogateCache = preinitializeBackpropagationCache(input.getRowCount());
			backpropogate(expected, predictedCache, backpropogateCache);

			// Update weights and biases for each layer
			for (size_t layer = 0; layer < layerCount - 1; layer++)
			{
				// Average weight deltas across each data point and include learning rate
				Matrix derivativeDelta = std::move(backpropogateCache.pdToWeights[layer][0]);
				Matrix biasDelta = std::move(backpropogateCache.pdToBias[layer][0]);
				for (size_t input = 1; input < backpropogateCache.pdToWeights[layer].size(); input++)
				{
					derivativeDelta += backpropogateCache.pdToWeights[layer][input];
					biasDelta += backpropogateCache.pdToBias[layer][input];
				}
				derivativeDelta *= -config.learningRate / backpropogateCache.pdToWeights[layer].size();
				biasDelta *= -config.learningRate / backpropogateCache.pdToWeights[layer].size();

				// Carry forward weight momentum into delta
				if (!weightsMomentum[layer].getEmpty())
				{
					weightsMomentum[layer] *= config.momentumRate;
					biasMomentum[layer] *= config.momentumRate;
					derivativeDelta += weightsMomentum[layer];
					biasDelta += biasMomentum[layer];
				}
				weightsMomentum[layer] = derivativeDelta;
				biasMomentum[layer] = biasDelta;

				// Apply weights gradient descent delta
				weights[layer] += derivativeDelta;
				bias[layer] += biasDelta;
			}

			// Return error
			return errorFn(predictedCache.neuronOutput[layerCount - 1], expected);
		}

		void SupervisedNetwork::backpropogate(const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const
		{
			// For each layer of weights
			for (size_t layer = 0; layer < layerCount - 1; layer++)
			{
				// Precalculate derivative of error w.r.t. to neuron in for next layer
				calculatePdErrorToIn(layer + 1, expected, predictedCache, backpropogateCache);
				Matrix const& neuronOut = predictedCache.neuronOutput[layer];
				Matrix const& pdNextNeuronIn = backpropogateCache.pdToNeuronIn[layer + 1];

				// For each input we are processing
				for (size_t input = 0; input < expected.getRowCount(); input++)
				{
					// Partial derivative of error w.r.t. to weight
					// (δE / δWᵢⱼ) = (δE / δzⱼ) * (δzⱼ / δWᵢⱼ)
					backpropogateCache.pdToWeights[layer][input] = Matrix(layerSizes[layer], layerSizes[layer + 1]);
					for (size_t i = 0; i < layerSizes[layer]; i++)
					{
						for (size_t j = 0; j < layerSizes[layer + 1]; j++)
						{
							backpropogateCache.pdToWeights[layer][input](i, j) = neuronOut(input, i) * pdNextNeuronIn(input, j);
						}
					}

					// Partial derivative of error w.r.t. to bias
					// (δE / δBᵢⱼ) = (δE / δzⱼ) * (δzⱼ / δBᵢⱼ)
					backpropogateCache.pdToBias[layer][input] = Matrix(1, layerSizes[layer + 1]);
					for (size_t j = 0; j < layerSizes[layer + 1]; j++)
					{
						backpropogateCache.pdToBias[layer][input](0, j) = pdNextNeuronIn(input, j);
					}
				}
			}
		}

		Matrix const& SupervisedNetwork::calculatePdErrorToIn(size_t layer, const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const
		{
			if (!backpropogateCache.pdToNeuronIn[layer].getEmpty()) return backpropogateCache.pdToNeuronIn[layer];

			// Partial derivative of error w.r.t. to neuron in
			// (δE / δzⱼ) = Σ(δE / δoᵢ) * (δoᵢ / δzⱼ)
			const Matrix& neuronOutputs = predictedCache.neuronOutput[layer];
			const Matrix& pdToOut = calculatePdErrorToOut(layer, expected, predictedCache, backpropogateCache);
			backpropogateCache.pdToNeuronIn[layer] = actFns[layer - 1].derivative(neuronOutputs, pdToOut);

			return backpropogateCache.pdToNeuronIn[layer];
		}

		Matrix const& SupervisedNetwork::calculatePdErrorToOut(size_t layer, const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const
		{
			if (!backpropogateCache.pdToNeuronOut[layer].getEmpty()) return backpropogateCache.pdToNeuronOut[layer];

			// Partial derivative of next layer in w.r.t current layer out
			// (δE / δoⱼ) = Σ(δWᵢⱼ * δₗ)
			else if (layer < layerCount - 1)
			{
				Matrix weightsT = weights[layer].transposed();
				const Matrix& pdToNextIn = calculatePdErrorToIn(layer + 1, expected, predictedCache, backpropogateCache);
				backpropogateCache.pdToNeuronOut[layer] = pdToNextIn.crossed(weightsT);
			}

			// Partial derivative of error w.r.t. to neuron out
			// (δE / δoⱼ) = (δE / δy)
			else if (layer == layerCount - 1)
			{
				const Matrix& neuronOutputs = predictedCache.neuronOutput[layer];
				backpropogateCache.pdToNeuronOut[layer] = errorFn.derivative(neuronOutputs, expected);
			}

			return backpropogateCache.pdToNeuronOut[layer];
		}

		BackpropogateCache SupervisedNetwork::preinitializeBackpropagationCache(int inputCount) const
		{
			BackpropogateCache cache;

			cache.pdToWeights = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights
			cache.pdToBias = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = bias
			cache.pdToNeuronOut = std::vector<Matrix>(layerCount); // element = layer, row = input, column = neuron
			cache.pdToNeuronIn = std::vector<Matrix>(layerCount); // element = layer, row = input, column = neuron

			for (size_t layer = 0; layer < layerCount - 1; layer++)
			{
				cache.pdToWeights[layer] = std::vector<Matrix>(inputCount);
				cache.pdToBias[layer] = std::vector<Matrix>(inputCount);
			}

			return cache;
		}
	}
}
