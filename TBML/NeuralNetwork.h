
#pragma once

#include "Matrix.h"
#include "UtilityFunctions.h"

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

		std::vector<Matrix>& getWeights();
		std::vector<Matrix>& getBias();
		const std::vector<Matrix>& getWeights() const;
		const std::vector<Matrix>& getBias() const;
		afptr getActivator() const;
		size_t getLayerCount() const;
		std::vector<size_t> getLayerSizes() const;
		size_t getInputSize() const;

	protected:
		size_t layerCount;
		std::vector<size_t> layerSizes;
		std::vector<Matrix> weights;
		std::vector<Matrix> bias;
		float (*activator)(float);
	};
}
