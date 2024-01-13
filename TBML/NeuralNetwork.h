#pragma once

#include "Matrix.h"
#include "Utility.h"

namespace tbml
{
	struct PropogateCache
	{
		std::vector<Matrix> neuronOutput;
	};

	class NeuralNetwork
	{
	public:
		enum WeightInitType { ZERO, RANDOM };

		NeuralNetwork() {}
		NeuralNetwork(std::vector<size_t> layerSizes, WeightInitType weightInitType = RANDOM);
		NeuralNetwork(std::vector<size_t> layerSizes, std::vector<fns::ActivationFunction> actFns, WeightInitType weightInitType = RANDOM);
		NeuralNetwork(std::vector<Matrix> weights, std::vector<Matrix> bias, std::vector<fns::ActivationFunction> actFns);

		void InitializeWeights(WeightInitType type);
		Matrix propogate(const Matrix& input) const;
		void propogate(const Matrix& input, PropogateCache& cache) const;
		void printLayers() const;

		std::vector<Matrix>& getWeights() { return weights; }
		std::vector<Matrix>& getBias() { return bias; }
		const std::vector<Matrix>& getWeights() const { return weights; }
		const std::vector<Matrix>& getBias() const { return bias; }
		std::vector<fns::ActivationFunction> getActivationFns() const { return actFns; }
		size_t getLayerCount() const { return layerCount; }
		std::vector<size_t> getLayerSizes() const { return layerSizes; }
		size_t getInputSize() const { return layerSizes[0]; }

	protected:
		size_t layerCount = 0;
		std::vector<size_t> layerSizes;
		std::vector<Matrix> weights;
		std::vector<Matrix> bias;
		std::vector<fns::ActivationFunction> actFns;
	};
}
