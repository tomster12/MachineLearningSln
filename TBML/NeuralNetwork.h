
#pragma once

#include "Matrix.h"
#include "Utility.h"

namespace tbml
{
	struct PropogateCache
	{
		std::vector<Matrix> neuronOutput;
	};

	typedef float (*afptr)(float);

	class NeuralNetwork
	{
	public:
		NeuralNetwork() {}
		NeuralNetwork(std::vector<size_t> layerSizes, float (*activator)(float) = tbml::sigmoid, bool randomize = true);
		NeuralNetwork(std::vector<Matrix> weights, std::vector<Matrix> bias, float (*activator)(float) = tbml::sigmoid);

		void randomize();
		Matrix propogate(const Matrix& input) const;
		void propogate(const Matrix& input, PropogateCache& cache) const;
		void printLayers() const;

		std::vector<Matrix>& getWeights() { return weights; }
		std::vector<Matrix>& getBias() { return bias; }
		const std::vector<Matrix>& getWeights() const { return weights; }
		const std::vector<Matrix>& getBias() const { return bias; }
		afptr getActivator() const { return activator; }
		size_t getLayerCount() const { return layerCount; }
		std::vector<size_t> getLayerSizes() const { return layerSizes; }
		size_t getInputSize() const { return layerSizes[0]; }

	protected:
		size_t layerCount = 0;
		std::vector<size_t> layerSizes;
		std::vector<Matrix> weights;
		std::vector<Matrix> bias;
		float (*activator)(float) = nullptr;
	};
}
