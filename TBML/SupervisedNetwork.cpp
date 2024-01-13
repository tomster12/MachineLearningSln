#include "stdafx.h"
#include "SupervisedNetwork.h"
#include "Utility.h"
#include "Matrix.h"
#include "ThreadPool.h"

namespace tbml
{
	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes, WeightInitType weightInitType)
		: SupervisedNetwork(layerSizes, fns::SquareError(), weightInitType)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes, std::vector<fns::ActivationFunction> actFns, WeightInitType weightInitType)
		: SupervisedNetwork(layerSizes, actFns, fns::SquareError(), weightInitType)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes, fns::ErrorFunction errorFn, WeightInitType weightInitType)
		: NeuralNetwork(layerSizes, weightInitType), errorFn(errorFn)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes, std::vector<fns::ActivationFunction> actFns, fns::ErrorFunction errorFn, WeightInitType weightInitType)
		: NeuralNetwork(layerSizes, actFns, weightInitType), errorFn(errorFn)
	{}

	void SupervisedNetwork::train(const Matrix& input, const Matrix& expected, const TrainingConfig& config)
	{
		size_t batchCount;
		std::vector<Matrix> batchInputs, batchExpected;

		// Pull data and expected into a single batch
		if (config.batchSize == -1)
		{
			batchInputs = std::vector<Matrix>({ input });
			batchExpected = std::vector<Matrix>({ expected });
			batchCount = 1;
		}

		else
			// Split input and expected into batches
		{
			batchInputs = input.groupRows(config.batchSize);
			batchExpected = expected.groupRows(config.batchSize);
			batchCount = config.batchSize;
		}

		// Initialize training loop variables
		std::mutex updateMutex;
		std::vector<Matrix> weightsMomentum = std::vector<Matrix>(layerCount);
		std::vector<Matrix> biasMomentum = std::vector<Matrix>(layerCount);
		int maxEpochs = (config.epochs == -1 && config.errorExit > 0.0f) ? MAX_MAX_ITERATIONS : config.epochs;
		int epoch = 0;

		//ThreadPool threadPool;
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point tepoch = t0;
		for (; epoch < maxEpochs; epoch++)
		{
			float epochError = 0.0f;

			// Process each batch
			//std::vector<std::future<float>> results(batchCount);
			for (size_t batch = 0; batch < batchCount; batch++)
			{
				const Matrix& input = batchInputs[batch];
				const Matrix& expected = batchExpected[batch];
				//results[batch] = threadPool.enqueue([&, batch]
				//{
				std::chrono::steady_clock::time_point tbatch = std::chrono::steady_clock::now();

				float batchError = trainBatch(input, expected, config, weightsMomentum, biasMomentum, updateMutex);
				epochError += batchError;

				std::chrono::steady_clock::time_point tnow = std::chrono::steady_clock::now();
				auto us = std::chrono::duration_cast<std::chrono::microseconds>(tnow - tbatch);
				std::cout << "Epoch: " << epoch << ", Batch: " << batch << ", batch time: " << us.count() / 1000 << "ms | Batch Error: " << batchError << std::endl;
				//return batchError;
				//});
			}
			//for (auto& result : results) epochError += result.get();

			// Log curret epoch
			if (config.logLevel >= 2)
			{
				std::chrono::steady_clock::time_point tnow = std::chrono::steady_clock::now();
				auto us = std::chrono::duration_cast<std::chrono::microseconds>(tnow - tepoch);
				std::cout << "Epoch: " << epoch << ", epoch time: " << us.count() / 1000 << "ms | Epoch Error: " << epochError << std::endl;
				tepoch = tnow;
			}

			// Exit early if error below certain amount
			if (epochError < config.errorExit) { epoch++; break; }
		}

		// Print training outcome
		if (config.logLevel >= 1)
		{
			std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
			auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
			std::cout << std::endl << "-- Finished training --" << std::endl;
			std::cout << "Epochs: " << epoch << std::endl;
			std::cout << "Time taken: " << us.count() / 1000 << "ms" << std::endl << std::endl;
		}
	}

	float SupervisedNetwork::trainBatch(const Matrix& input, const Matrix& expected, const TrainingConfig& config, std::vector<Matrix>& weightsMomentum, std::vector<Matrix>& biasMomentum, std::mutex& updateMutex)
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

			// Apply weight gradient clipping
			// TODO

			// Apply weights delta
			{
				std::lock_guard<std::mutex> guard(updateMutex);
				weights[layer] += derivativeDelta;
				bias[layer] += biasDelta;
			}
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
				// (δE / δWᵢⱼ) = (δE / δnetⱼ) * (δnetⱼ / δWᵢⱼ)
				//            = (δE / δnetⱼ) * (oᵢ)
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
				// (δE / δBᵢⱼ) = (δE / δnetⱼ) * (δnetⱼ / δBᵢⱼ)
				//            = (δE / δnetⱼ)
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

		calculatePdErrorToOut(layer, expected, predictedCache, backpropogateCache);
		const Matrix& pdToOut = backpropogateCache.pdNeuronOut[layer];

		// Partial derivative of error w.r.t. to neuron in
		// (δE / δnetⱼ) = (δE / δoⱼ) * (δoⱼ / δnetᵢⱼ)
		backpropogateCache.pdNeuronIn[layer] = actFns[layer - 1].derive(predictedCache.neuronOutput[layer]);
		backpropogateCache.pdNeuronIn[layer] *= pdToOut;
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
			// TODO: This value is becoming enourmous
			backpropogateCache.pdNeuronOut[layerCount - 1] = errorFn.derive(predictedCache.neuronOutput[layerCount - 1], expected);
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

// --------------------------------
//
// 	-- Forward Propogation --
//
// - Overview -
//
// Layer Sizes:  2, 4, 1
// Bias:		 True
// Data Count:	 3
//
// - Data Layout -
//
// Input	= [ x00, x01 ]
//			  [ x10, x11 ]
//			  [ x20, x21 ]
//
//
// weights l1 = [ w00 w01 w02 w03 ]
//			    [ w10 w11 w12 w13 ]
//
// bias l1	  = [ b0, b1, b2, b3 ]
//
// Values l1  = [ x00, x01, x02, x03 ]
//				[ x10, x11, x12, x13 ]
//				[ x20, x21, x22, x23 ]
//
//
// weight l2  = [ w00 ]
//			    [ w10 ]
//			    [ w20 ]
//			    [ w30 ]
//
// bias l2	  = [ b0 ]
//
// Values l2  = [ x00 ]
//				[ x10 ]
//				[ x20 ]
//
// --------------------------------

// --------------------------------
//
// 	-- Back Propogation --
//
// (δE / δWᵢⱼ)		= (δE / δoⱼ) * (δoⱼ / δnetⱼ) * (δnetⱼ / δWᵢⱼ)
//
// pdErrorToWeight	= pdErrorToIn * pdInToWeight
// (δE / δWᵢⱼ)		= (δE / δnetⱼ) * (δnetⱼ / δWᵢⱼ)
//					= (δE / δnetⱼ) * Out[j]
//
// pdErrorToIn		= pdErrorToOut * pdOutToIn			(Cache)
// (δE / δnetⱼ)		= (δE / δoⱼ)   * (δoⱼ / δnetᵢⱼ)
//
// pdErrorToOut[last]	= predicted - expected
// (δE / δoⱼ)			= (δE / δy)
//						= (y - t)
//
// pdErrorToOut[other]	= sum(weight[J] * pdErrorToIn[J])
// (δE / δoⱼ)			= Σ(δWᵢⱼ * (δE / δnetⱼ))
//
// pdOutToIn
// (δoⱼ / δnetᵢⱼ)	= drvActivator(out[j])
//
// Error
// E(y)				= 1/2 * Σ(t - y) ^ 2
//
// -- Gradient Descent --
//
// weight += derivative * learningRate
// weight += momentum * momentumRate
// momentum = derivative
//
// --------------------------------
