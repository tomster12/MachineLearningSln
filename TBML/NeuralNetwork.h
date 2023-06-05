
#pragma once

#include "Matrix.h"
#include "UtilityFunctions.h"

namespace tbml
{
	typedef float (*afptr)(float);

	class NeuralNetwork
	{
	public:
		NeuralNetwork() {}
		NeuralNetwork(std::vector<size_t> layerSizes, float (*activator)(float) = tbml::sigmoid, bool randomize = true);
		NeuralNetwork(std::vector<Matrix> weights, std::vector<Matrix> bias, float (*activator)(float) = tbml::sigmoid);

		void randomize();
		Matrix& propogate(Matrix& input);
		void printLayers();

		float getCachedValue(int layer, int row, int col);
		std::vector<Matrix>& getWeights();
		std::vector<Matrix>& getBias();
		afptr getActivator();

	protected:
		size_t layerCount;
		std::vector<size_t> layerSizes;
		std::vector<Matrix> weights;
		std::vector<Matrix> bias;
		float (*activator)(float);
		std::vector<Matrix> neuronOutCache;
	};
}
