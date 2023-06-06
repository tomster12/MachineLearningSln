
#include "stdafx.h"
#include "SupervisedNetwork.h"
#include "UtilityFunctions.h"
#include "Matrix.h"
#include "ThreadPool.h"

namespace tbml
{
	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes_)
		: SupervisedNetwork(layerSizes_, tanh, tanhPd, calcErrSqDiff, calcErrSqDiffPd)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float))
		: SupervisedNetwork(layerSizes_, activator_, activatorPd_, calcErrSqDiff, calcErrSqDiffPd)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes_, float (*calcError_)(const Matrix&, const Matrix&), Matrix(*calcErrorPd_)(const Matrix&, const Matrix&))
		: SupervisedNetwork(layerSizes_, tanh, tanhPd, calcError_, calcErrorPd_)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float), float (*calcError_)(const Matrix&, const Matrix&), Matrix(*calcErrorPd_)(const Matrix&, const Matrix&))
		: NeuralNetwork(layerSizes_, activator_),
		activatorPd(activatorPd_),
		calcError(calcError_),
		calcErrorPd(calcErrorPd_)
	{}

	void SupervisedNetwork::train(const Matrix& input, const Matrix& expected, const TrainingConfig& config)
	{
		// Split input and expected
		size_t batchCount;
		std::vector<Matrix> splitInput, splitExpected;
		if (config.batchSize == -1)
		{
			splitInput = std::vector<Matrix>({ input });
			splitExpected = std::vector<Matrix>({ expected });
			batchCount = 1;
		}
		else
		{
			splitInput = input.splitRows(config.batchSize);
			splitExpected = expected.splitRows(config.batchSize);
			batchCount = splitInput.size();
		}

		// Setup variables
		std::mutex updateMutex;
		std::vector<Matrix> weightsMomentum = std::vector<Matrix>(layerCount);
		std::vector<Matrix> biasMomentum = std::vector<Matrix>(layerCount);
		int maxEpochs = (config.epochs == -1 && config.errorExit > 0.0f) ? MAX_MAX_ITERATIONS : config.epochs;
		int epoch = 0;

		// Main timed training loop
		//ThreadPool threadPool;
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point tepoch = t0;
		std::chrono::steady_clock::time_point tbatch = t0;
		for (; epoch < maxEpochs; epoch++)
		{
			float epochError = 0.0f;
			std::vector<std::future<float>> results(batchCount);
			for (size_t batch = 0; batch < batchCount; batch++)
			{
				const Matrix& input = splitInput[batch];
				const Matrix& expected = splitExpected[batch];
				//results[batch] = threadPool.enqueue([=]
				//{
				float batchError = trainBatch(input, expected, config, weightsMomentum, biasMomentum, updateMutex);
				epochError += batchError;
				std::chrono::steady_clock::time_point tnow = std::chrono::steady_clock::now();
				auto us = std::chrono::duration_cast<std::chrono::microseconds>(tnow - tbatch);
				std::cout << "Batch: " << batch << ", batch time: " << us.count() / 1000 << "ms | Batch Error: " << batchError << std::endl;
				tbatch = tnow;
				//	return batchError;
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
		backpropogate(expected, predictedCache, backpropogateCache);

		// Update weights and biases for each layer
		for (size_t layer = 0; layer < layerCount - 1; layer++)
		{
			// Calculate scaled average of derivatives
			Matrix derivativeDelta = backpropogateCache.pdWeights[layer][0];
			Matrix biasDelta = backpropogateCache.pdBias[layer][0];
			for (size_t input = 1; input < backpropogateCache.pdWeights[layer].size(); input++)
			{
				derivativeDelta.iadd(backpropogateCache.pdWeights[layer][input]);
				biasDelta.iadd(backpropogateCache.pdBias[layer][input]);
			}
			derivativeDelta.iscale(-config.learningRate / backpropogateCache.pdWeights[layer].size());
			biasDelta.iscale(-config.learningRate / backpropogateCache.pdWeights[layer].size());

			// Carry forward momentum
			{
				std::lock_guard<std::mutex> guard(updateMutex);

				if (!weightsMomentum[layer].getEmpty())
				{
					weightsMomentum[layer].iscale(config.momentumRate);
					biasMomentum[layer].iscale(config.momentumRate);
					derivativeDelta.iadd(weightsMomentum[layer]);
					biasDelta.iadd(biasMomentum[layer]);
				}
				weightsMomentum[layer] = derivativeDelta;
				biasMomentum[layer] = biasDelta;
				weights[layer].iadd(derivativeDelta);
				bias[layer].iadd(biasDelta);
			}
		}

		// Return error
		return calcError(predictedCache.neuronOutput[layerCount - 1], expected);
	}

	void SupervisedNetwork::backpropogate(const Matrix& expected, const PropogateCache& predictedCache, BackpropogateCache& backpropgateCache) const
	{
		// Reinitialize cache
		backpropgateCache.pdOut = Matrix(); // row = input, column = neuron
		if (backpropgateCache.pdWeights.size() != layerCount - 1)
		{
			backpropgateCache.pdNeuronOut = std::vector<Matrix>(layerCount); // element = layer, row = input, column = neuron
			backpropgateCache.pdNeuronIn = std::vector<Matrix>(layerCount); // element = layer, row = input, column = neuron
			backpropgateCache.pdWeights = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights
			backpropgateCache.pdBias = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights
		}
		else
		{
			for (size_t i = 0; i < layerCount; i++)
			{
				backpropgateCache.pdNeuronOut[i].clear();
				backpropgateCache.pdNeuronIn[i].clear();
			}
		}

		// Partial derivative of error w.r.t. to neuron out
		// (δE / δoⱼ) = (δE / δy)
		backpropgateCache.pdNeuronOut[layerCount - 1] = calcErrorPd(predictedCache.neuronOutput[layerCount - 1], expected);

		// Partial derivative of error w.r.t. to weight
		// (δE / δWᵢⱼ) = (δE / δnetⱼ) * (δnetⱼ / δWᵢⱼ)
		// - Loop over layers
		for (size_t layer = 0; layer < layerCount - 1; layer++)
		{
			calculatePdErrorToIn(layer + 1, predictedCache, backpropgateCache);
			const Matrix& neuronOut = predictedCache.neuronOutput[layer];
			Matrix& pdNeuronIn = backpropgateCache.pdNeuronIn[layer + 1];

			// - Loop over each input
			size_t inputCount = expected.getRows();
			if (backpropgateCache.pdWeights[layer].size() != inputCount) backpropgateCache.pdWeights[layer] = std::vector<Matrix>(inputCount);
			if (backpropgateCache.pdBias[layer].size() != inputCount) backpropgateCache.pdBias[layer] = std::vector<Matrix>(inputCount);
			for (size_t input = 0; input < expected.getRows(); input++)
			{
				// - Calculate weight derivatives
				std::vector<std::vector<float>> pdWeightData = std::vector<std::vector<float>>(layerSizes[layer]);
				for (size_t row = 0; row < layerSizes[layer]; row++)
				{
					pdWeightData[row] = std::vector<float>(layerSizes[layer + 1]);
					for (size_t col = 0; col < layerSizes[layer + 1]; col++)
					{
						float val = neuronOut.get(input, row) * pdNeuronIn.get(input, col);
						pdWeightData[row][col] = val;
					}
				}
				backpropgateCache.pdWeights[layer][input] = Matrix(pdWeightData);

				// - Calculate bias derivatives
				std::vector<std::vector<float>> pdBiasData = std::vector<std::vector<float>>(1);
				pdBiasData[0] = std::vector<float>(layerSizes[layer + 1]);
				for (size_t col = 0; col < layerSizes[layer + 1]; col++)
				{
					float val = pdNeuronIn.get(input, col);
					pdBiasData[0][col] = val;
				}
				backpropgateCache.pdBias[layer][input] = Matrix(pdBiasData);
			}
		}
	}

	void SupervisedNetwork::calculatePdErrorToIn(size_t layer, const PropogateCache& predictedCache, BackpropogateCache& backpropgateCache) const
	{
		// Already calculated
		if (!backpropgateCache.pdNeuronIn[layer].getEmpty()) return;

		// Partial derivative of error w.r.t. to neuron in
		// (δE / δnetⱼ) = (δE / δoⱼ) * (δoⱼ / δnetᵢⱼ)
		calculatePdErrorToOut(layer, predictedCache, backpropgateCache);
		Matrix& pdToOut = backpropgateCache.pdNeuronOut[layer];
		Matrix pdToIn = predictedCache.neuronOutput[layer].map(activatorPd);
		pdToIn.itimes(pdToOut);
		backpropgateCache.pdNeuronIn[layer] = pdToIn;
	}

	void SupervisedNetwork::calculatePdErrorToOut(size_t layer, const PropogateCache& predictedCache, BackpropogateCache& backpropgateCache) const
	{
		// Already calculated
		if (!backpropgateCache.pdNeuronOut[layer].getEmpty()) return;

		// Start / middle layer derivative of activator
		// (δE / δoⱼ) = Σ(δWᵢⱼ * δₗ)
		else if (layer < layerCount - 1)
		{
			calculatePdErrorToIn(layer + 1, predictedCache, backpropgateCache);
			Matrix wt = weights[layer].transpose();
			backpropgateCache.pdNeuronOut[layer] = backpropgateCache.pdNeuronIn[layer + 1].cross(wt);
		}

		// Last layer derivative should already be calculated
		else throw std::exception("calculatePdErrorToOut() for last layer should already be calculated.");
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