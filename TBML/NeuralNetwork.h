
#pragma once

#include "Matrix.h"


namespace tbml
{
	class NeuralNetwork
	{
	protected:
		size_t layerCount;
		std::vector<size_t> layerSizes;
		std::vector<Matrix> weights;
		std::vector<Matrix> bias;
		float (*activator)(float);
		std::vector<Matrix> neuronOutCache;

	public:
		NeuralNetwork() {}
		NeuralNetwork(std::vector<size_t> layerSizes);
		NeuralNetwork(std::vector<size_t> layerSizes, float (*activator)(float), bool randomize = true);
		NeuralNetwork(std::vector<Matrix> weights, std::vector<Matrix> bias);
		NeuralNetwork(std::vector<Matrix> weights, std::vector<Matrix> bias, float (*activator)(float));

		void randomize();
		Matrix propogate(Matrix input);
		void printLayers();

		std::vector<Matrix>& getWeights();
		std::vector<Matrix>& getBias();
	};
}
