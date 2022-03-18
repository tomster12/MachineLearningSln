
#include <iostream>
#include <chrono>
#include "SupervisedNetwork.h"
#include "Activators.h"


SupervisedNetwork::SupervisedNetwork(std::vector<int> layerSizes_)
	: SupervisedNetwork(layerSizes_, actTanh, actTanhPd) { }

SupervisedNetwork::SupervisedNetwork(std::vector<int> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float))
	: NeuralNetwork(layerSizes_, activator_) {
	
	// Initialize variables
	activatorPd = activatorPd_;
}


void SupervisedNetwork::train(Matrix input, Matrix expected, TrainingConfig config) {
	// Setup variables
	pdWeightsMomentum = std::vector<Matrix>(layerCount);
	pdBiasMomentum = std::vector<Matrix>(layerCount);
	int maxEpochs = (config.epochs == -1 && config.errorExit > 0.0f) ? MAX_ITERATIONS : config.epochs;
	int epoch = 0;
	float error = -1;

	// Split input and expected
	size_t batchCount = 1;
	std::vector<Matrix> splitInput, splitExpected;
	if (config.batchSize == -1) {
		splitInput = std::vector<Matrix>(1);
		splitInput[0] = input;
		splitExpected = std::vector<Matrix>(1);
		splitExpected[0] = expected;
	} else {
		splitInput = input.splitRows(config.batchSize);
		splitExpected = expected.splitRows(config.batchSize);
		batchCount = splitInput.size();
	}

	// Start timer
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();


	// Loop over and run training
	for (; epoch < maxEpochs; epoch++) {

		// Forward / backwards propogate for current batch
		for (size_t batch = 0; batch < batchCount; batch++) {

			// Forward propogation current batch
			Matrix predicted = propogate(splitInput[batch]);
			error = calculateError(predicted, splitExpected[batch]);
			if (config.logLevel >= 2) std::cout << "Epoch: " << epoch << ", batch: " << batch << " | Error: " << error << std::endl;

			// Backwards propogate using batch gradient descent
			calculateDerivates(predicted, splitExpected[batch]);
			for (size_t layer = 0; layer < layerCount - 1; layer++) {

				// Calculate average derivative for weight / bias
				Matrix derivativeSum = pdWeightsCache[layer][0];
				Matrix biasSum = pdBiasCache[layer][0];
				for (size_t input = 1; input < predicted.getRows(); input++) {
					derivativeSum.iadd(pdWeightsCache[layer][input]);
					biasSum.iadd(pdBiasCache[layer][input]);
				}
				derivativeSum.iscale(1.0f / predicted.getRows());
				biasSum.iscale(1.0f / predicted.getRows());

				// Update weight / bias values
				weights[layer].iadd(derivativeSum.scale(-config.learningRate));
				bias[layer].iadd(biasSum.scale(-config.learningRate));
				if (epoch > 0) {
					pdWeightsMomentum[layer].iscale(-config.momentumRate);
					pdBiasMomentum[layer].iscale(-config.momentumRate);
					weights[layer].iadd(pdWeightsMomentum[layer]);
					bias[layer].iadd(pdBiasMomentum[layer]);
				}
				pdWeightsMomentum[layer] = derivativeSum;
				pdBiasMomentum[layer] = biasSum;
			}

			// Exit early if error below certain amount
			if (error < config.errorExit) break;
		}
	
		// Exit early if error below certain amount
		if (error < config.errorExit) break;
	}


	// Print training outcome
	if (config.logLevel >= 1) {
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
		std::cout << std::endl << "-- Finished training --" << std::endl;
		std::cout << "Epochs: " << epoch << std::endl;
		std::cout << "Final Error: " << error << std::endl;
		std::cout << "Time taken: " << us.count() / 1000 << "ms" << std::endl << std::endl;
	}
}


float SupervisedNetwork::calculateError(Matrix predicted, Matrix expected) {
	// Calculate squared sum error TODO: Unhardcode
	std::vector<std::vector<float>>* pData = predicted.getData();
	std::vector<std::vector<float>>* eData = expected.getData();
	float error = 0;
	for (int row = 0; row < predicted.getRows(); row++) {
		for (int col = 0; col < predicted.getCols(); col++) {
			float diff = (*eData)[row][col] - (*pData)[row][col];
			error += diff * diff;
		}
	}
	return 0.5f * error;
}


void SupervisedNetwork::calculateDerivates(Matrix predicted, Matrix expected) {
	// Reinitialize partial derivatives
	pdOutCache = Matrix(); // row = input, column = neuron
	pdNeuronInCache = std::vector<Matrix>(layerCount); // element = layer, row = input, column = neuron
	pdWeightsCache = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights
	pdBiasCache = std::vector<std::vector<Matrix>>(layerCount - 1); // element^1 = layer, element^2 = input, matrix = weights

	// Partial derivative of error w.r.t. to neuron out TODO: Unhardcode
	// (δE / δoⱼ) = (δE / δy)
	pdOutCache = predicted.sub(expected);

	// Partial derivative of error w.r.t. to weight
	// (δE / δWᵢⱼ) = (δE / δnetⱼ) * (δnetⱼ / δWᵢⱼ)
	// - Loop over layers
	for (size_t layer = 0; layer < layerCount - 1; layer++) {
		calculatePdErrorToIn(layer + 1);
		Matrix neuronOut = neuronOutCache[layer];
		Matrix pdNeuronIn = pdNeuronInCache[layer + 1];

		// - Loop over each input
		pdWeightsCache[layer] = std::vector<Matrix>();
		pdBiasCache[layer] = std::vector<Matrix>();
		for (size_t input = 0; input < expected.getRows(); input++) {
			
			// - Calculate weight derivatives
			std::vector<std::vector<float>> pdWeightData;
			for (size_t row = 0; row < layerSizes[layer]; row++) {
				pdWeightData.push_back(std::vector<float>());
				for (size_t col = 0; col < layerSizes[layer + 1]; col++) {
					float val = neuronOut.get(input, row) * pdNeuronIn.get(input, col);
					pdWeightData[row].push_back(val);
				}
			}
			pdWeightsCache[layer].push_back(Matrix(pdWeightData));

			// - Calculate bias derivatives
			std::vector<std::vector<float>> pdBiasData;
			pdBiasData.push_back(std::vector<float>());
			for (size_t col = 0; col < layerSizes[layer + 1]; col++) {
				float val = pdNeuronIn.get(input, col);
				pdBiasData[0].push_back(val);
			}
			pdBiasCache[layer].push_back(Matrix(pdBiasData));
		}
	}
}


void SupervisedNetwork::calculatePdErrorToIn(size_t layer) {
	// Already calculated
	if (!pdNeuronInCache[layer].getEmpty()) return;
	
	// Partial derivative of error w.r.t. to neuron in
	// (δE / δnetⱼ) = (δE / δoⱼ) * (δoⱼ / δnetᵢⱼ)
 	Matrix pdToOut = pdErrorToOut(layer);
	Matrix pdToIn = neuronOutCache[layer].map(activatorPd);
	pdNeuronInCache[layer] = *pdToOut.itimes(pdToIn);
}


Matrix SupervisedNetwork::pdErrorToOut(size_t layer) {
	// Last layer derivative of error
	// (δE / δoⱼ) = (δE / δy)
	if (layer == layerCount - 1) return pdOutCache;
	
	// Start / middle layer derivative of activator
	// (δE / δoⱼ) = Σ(δWᵢⱼ * δₗ)
	else {
		calculatePdErrorToIn(layer + 1);
		return pdNeuronInCache[layer + 1].cross(weights[layer].transpose());
	}
}
