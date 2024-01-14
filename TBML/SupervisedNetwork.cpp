#include <iomanip>
#include "stdafx.h"
#include "SupervisedNetwork.h"
#include "Utility.h"
#include "Matrix.h"
#include "ThreadPool.h"

namespace tbml
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
		BackpropogateCache backpropogateCache;
		InitializeBackpropagateCache(backpropogateCache);
		backpropogate(expected, predictedCache, backpropogateCache);

		// Update weights and biases for each layer
		for (size_t layer = 0; layer < layerCount - 1; layer++)
		{
			// Average weight deltas across each data point and include learning rate
			Matrix derivativeDelta = std::move(backpropogateCache.pdWeights[layer][0]);
			Matrix biasDelta = std::move(backpropogateCache.pdBias[layer][0]);
			for (size_t input = 1; input < backpropogateCache.pdWeights[layer].size(); input++)
			{
				derivativeDelta += backpropogateCache.pdWeights[layer][input];
				biasDelta += backpropogateCache.pdBias[layer][input];
			}
			derivativeDelta *= -config.learningRate / backpropogateCache.pdWeights[layer].size();
			biasDelta *= -config.learningRate / backpropogateCache.pdWeights[layer].size();

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

			// Apply weights delta
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
			const Matrix& neuronOut = predictedCache.neuronOutput[layer];
			const Matrix& pdNextNeuronIn = backpropogateCache.pdNeuronIn[layer + 1];

			// For each input we are processing
			size_t inputCount = expected.getRowCount();
			if (backpropogateCache.pdWeights[layer].size() != inputCount) backpropogateCache.pdWeights[layer] = std::vector<Matrix>(inputCount);
			if (backpropogateCache.pdBias[layer].size() != inputCount) backpropogateCache.pdBias[layer] = std::vector<Matrix>(inputCount);
			for (size_t input = 0; input < expected.getRowCount(); input++)
			{
				// Partial derivative of error w.r.t. to weight
				// (δE / δWᵢⱼ) = (δE / δzⱼ) * (δzⱼ / δWᵢⱼ)
				std::vector<float> pdWeights = std::vector<float>(layerSizes[layer] * layerSizes[layer + 1]);
				for (size_t row = 0; row < layerSizes[layer]; row++)
				{
					for (size_t col = 0; col < layerSizes[layer + 1]; col++)
					{
						pdWeights[row * layerSizes[layer + 1] + col] = neuronOut(input, row) * pdNextNeuronIn(input, col);
					}
				}
				backpropogateCache.pdWeights[layer][input] = Matrix(std::move(pdWeights), layerSizes[layer], layerSizes[layer + 1]);

				// Partial derivative of error w.r.t. to bias
				// (δE / δBᵢⱼ) = (δE / δzⱼ) * (δzⱼ / δBᵢⱼ)
				std::vector<float> pdBias = std::vector<float>(layerSizes[layer + 1]);
				for (size_t col = 0; col < layerSizes[layer + 1]; col++)
				{
					pdBias[col] = pdNextNeuronIn(input, col);
				}
				backpropogateCache.pdBias[layer][input] = Matrix(std::move(pdBias), 1, layerSizes[layer + 1]);
			}
		}
	}

	void SupervisedNetwork::calculatePdErrorToIn(size_t layer, const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const
	{
		if (!backpropogateCache.pdNeuronIn[layer].getEmpty()) return;

		// Partial derivative of error w.r.t. to neuron in
		// (δE / δzⱼ) = Σ(δE / δoᵢ) * (δoᵢ / δzⱼ)
		calculatePdErrorToOut(layer, expected, predictedCache, backpropogateCache);
		const Matrix& pdToOut = backpropogateCache.pdNeuronOut[layer];
		backpropogateCache.pdNeuronIn[layer] = actFns[layer - 1].derivative(predictedCache.neuronOutput[layer], pdToOut);
	}

	void SupervisedNetwork::calculatePdErrorToOut(size_t layer, const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropogateCache) const
	{
		if (!backpropogateCache.pdNeuronOut[layer].getEmpty()) return;

		// Partial derivative of next layer in w.r.t current layer out
		// (δE / δoⱼ) = Σ(δWᵢⱼ * δₗ)
		else if (layer < layerCount - 1)
		{
			calculatePdErrorToIn(layer + 1, expected, predictedCache, backpropogateCache);
			Matrix wt = weights[layer].transposed();
			backpropogateCache.pdNeuronOut[layer] = backpropogateCache.pdNeuronIn[layer + 1].crossed(wt);
		}

		// Partial derivative of error w.r.t. to neuron out
		// (δE / δoⱼ) = (δE / δy)
		else if (layer == layerCount - 1)
		{
			backpropogateCache.pdNeuronOut[layerCount - 1] = errorFn.derivative(predictedCache.neuronOutput[layerCount - 1], expected);
		}
	}

	void SupervisedNetwork::InitializeBackpropagateCache(BackpropogateCache& backpropogateCache) const
	{
		backpropogateCache.pdOut = Matrix(); // row = input, column = neuron
		if (backpropogateCache.pdWeights.size() != layerCount - 1)
		{
			backpropogateCache.pdNeuronOut = std::vector<Matrix>(layerCount); // element = layer, row = input, column = neuron
			backpropogateCache.pdNeuronIn = std::vector<Matrix>(layerCount); // element = layer, row = input, column = neuron
			backpropogateCache.pdWeights = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights
			backpropogateCache.pdBias = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights
		}
		else
		{
			for (size_t i = 0; i < layerCount; i++)
			{
				backpropogateCache.pdNeuronOut[i].clear();
				backpropogateCache.pdNeuronIn[i].clear();
			}
		}
	}
}
