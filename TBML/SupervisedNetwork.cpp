
#include "stdafx.h"
#include "SupervisedNetwork.h"
#include "UtilityFunctions.h"
#include "Matrix.h"


namespace tbml
{
	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes_)
		: SupervisedNetwork(layerSizes_, tanh, tanhPd, calcErrSqDiff, calcErrSqDiffPd)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float))
		: SupervisedNetwork(layerSizes_, activator_, activatorPd_, calcErrSqDiff, calcErrSqDiffPd)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes_, float (*calcError_)(Matrix, Matrix), Matrix(*calcErrorPd_)(Matrix, Matrix))
		: SupervisedNetwork(layerSizes_, tanh, tanhPd, calcError_, calcErrorPd_)
	{}

	SupervisedNetwork::SupervisedNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float), float (*calcError_)(Matrix, Matrix), Matrix(*calcErrorPd_)(Matrix, Matrix))
		: NeuralNetwork(layerSizes_, activator_)
	{

		// Initialize variables
		activatorPd = activatorPd_;
		calcError = calcError_;
		calcErrorPd = calcErrorPd_;
	}


	void SupervisedNetwork::train(Matrix input, Matrix expected, TrainingConfig config)
	{
		// Setup variables
		pdWeightsMomentum = std::vector<Matrix>(layerCount);
		pdBiasMomentum = std::vector<Matrix>(layerCount);
		int maxEpochs = (config.epochs == -1 && config.errorExit > 0.0f) ? MAX_maxIterations : config.epochs;
		int epoch = 0;
		float error = 0;

		// Split input and expected
		size_t batchCount = 1;
		std::vector<Matrix> splitInput, splitExpected;
		if (config.batchSize == -1)
		{
			splitInput = std::vector<Matrix>(1);
			splitInput[0] = input;
			splitExpected = std::vector<Matrix>(1);
			splitExpected[0] = expected;
		} else
		{
			splitInput = input.splitRows(config.batchSize);
			splitExpected = expected.splitRows(config.batchSize);
			batchCount = splitInput.size();
		}


		// Start timer
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point tmid = t0;


		// Loop over and run training
		for (; epoch < maxEpochs; epoch++)
		{

			// Forward / backwards propogate for current batch
			error = 0;
			for (size_t batch = 0; batch < batchCount; batch++)
			{

				// Forward propogation current batch
				Matrix predicted = propogate(splitInput[batch]);
				float batchError = calcError(predicted, splitExpected[batch]);
				error += batchError / batchCount;
				if (config.logLevel >= 2)
				{
					std::chrono::steady_clock::time_point tnow = std::chrono::steady_clock::now();
					auto us = std::chrono::duration_cast<std::chrono::microseconds>(tnow - tmid);
					if (batchCount == 1) std::cout << "Epoch: " << epoch << ", epoch time: " << us.count() / 1000 << "ms | Error: " << batchError << std::endl;
					else std::cout << "Epoch: " << epoch << ", batch: " << batch << ", batch time: " << us.count() / 1000 << "ms | Batch Error: " << batchError << std::endl;
					tmid = tnow;
				}

				// Backwards propogate using mini-batch gradient descent
				calculateDerivates(predicted, splitExpected[batch]);
				for (size_t layer = 0; layer < layerCount - 1; layer++)
				{

					// Calculate average derivative for weights / bias
					Matrix derivativeSum = pdWeightsCache[layer][0];
					Matrix biasSum = pdBiasCache[layer][0];
					for (size_t input = 1; input < predicted.getRows(); input++)
					{
						derivativeSum.iadd(pdWeightsCache[layer][input]);
						biasSum.iadd(pdBiasCache[layer][input]);
					}
					derivativeSum.iscale(-config.learningRate / predicted.getRows());
					biasSum.iscale(-config.learningRate / predicted.getRows());

					// Add momentum to weight / bias delta
					if (epoch > 0 || batch > 0)
					{
						pdWeightsMomentum[layer].iscale(config.momentumRate);
						pdBiasMomentum[layer].iscale(config.momentumRate);
						derivativeSum.iadd(pdWeightsMomentum[layer]);
						biasSum.iadd(pdBiasMomentum[layer]);
					}

					// Update weights / bias / momentum
					weights[layer].iadd(derivativeSum);
					bias[layer].iadd(biasSum);

					pdWeightsMomentum[layer] = derivativeSum;
					pdBiasMomentum[layer] = biasSum;
				}
			}

			// Exit early if error below certain amount
			if (config.logLevel >= 2 && batchCount != 1) std::cout << "Epoch Error: " << error << std::endl;
			if (error < config.errorExit) { epoch++; break; }
		}


		// Print training outcome
		if (config.logLevel >= 1)
		{
			std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
			auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
			std::cout << std::endl << "-- Finished training --" << std::endl;
			std::cout << "Epochs: " << epoch << std::endl;
			std::cout << "Average Error: " << error << std::endl;
			std::cout << "Time taken: " << us.count() / 1000 << "ms" << std::endl << std::endl;
		}
	}


	void SupervisedNetwork::calculateDerivates(Matrix& predicted, Matrix& expected)
	{
		// Reinitialize partial derivative caches
		pdOutCache = Matrix(); // row = input, column = neuron
		if (pdWeightsCache.size() != layerCount - 1)
		{
			pdNeuronInCache = std::vector<Matrix>(layerCount); // element = layer, row = input, column = neuron
			pdWeightsCache = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights
			pdBiasCache = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights
		} else
		{
			for (size_t i = 0; i < layerCount; i++) pdNeuronInCache[i].clear();
		}

		// Partial derivative of error w.r.t. to neuron out
		// (δE / δoⱼ) = (δE / δy)
		pdOutCache = calcErrorPd(predicted, expected);

		// Partial derivative of error w.r.t. to weight
		// (δE / δWᵢⱼ) = (δE / δnetⱼ) * (δnetⱼ / δWᵢⱼ)
		// - Loop over layers
		for (size_t layer = 0; layer < layerCount - 1; layer++)
		{
			calculatePdErrorToIn(layer + 1);
			Matrix& neuronOut = neuronOutCache[layer];
			Matrix& pdNeuronIn = pdNeuronInCache[layer + 1];

			// - Loop over each input
			size_t inputCount = expected.getRows();
			if (pdWeightsCache[layer].size() != inputCount) pdWeightsCache[layer] = std::vector<Matrix>(inputCount);
			if (pdBiasCache[layer].size() != inputCount) pdBiasCache[layer] = std::vector<Matrix>(inputCount);
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
				pdWeightsCache[layer][input] = Matrix(pdWeightData);

				// - Calculate bias derivatives
				std::vector<std::vector<float>> pdBiasData = std::vector<std::vector<float>>(1);
				pdBiasData[0] = std::vector<float>(layerSizes[layer + 1]);
				for (size_t col = 0; col < layerSizes[layer + 1]; col++)
				{
					float val = pdNeuronIn.get(input, col);
					pdBiasData[0][col] = val;
				}
				pdBiasCache[layer][input] = Matrix(pdBiasData);
			}
		}
	}

	void SupervisedNetwork::calculatePdErrorToIn(size_t layer)
	{
		// Already calculated
		if (!pdNeuronInCache[layer].getEmpty()) return;

		// Partial derivative of error w.r.t. to neuron in
		// (δE / δnetⱼ) = (δE / δoⱼ) * (δoⱼ / δnetᵢⱼ)
		Matrix pdToOut = pdErrorToOut(layer);
		Matrix pdToIn = neuronOutCache[layer].map(activatorPd);
		pdNeuronInCache[layer] = *pdToOut.itimes(pdToIn);
	}

	Matrix SupervisedNetwork::pdErrorToOut(size_t layer)
	{
		// Last layer derivative of error
		// (δE / δoⱼ) = (δE / δy)
		if (layer == layerCount - 1) return pdOutCache;

		// Start / middle layer derivative of activator
		// (δE / δoⱼ) = Σ(δWᵢⱼ * δₗ)
		else
		{
			calculatePdErrorToIn(layer + 1);
			Matrix wt = weights[layer].transpose();
			Matrix pdErrorToOut = pdNeuronInCache[layer + 1].cross(wt);
			return pdErrorToOut;
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
// pdErrorToWeight	= pdErrorToIn  * pdInToWeight
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
// 
// -- Gradient Descent --
// 
// weight += derivative * learningRate
// weight += momentum * momentumRate
// momentum = derivative
// 
// --------------------------------