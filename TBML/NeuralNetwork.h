
#pragma once

#include "Matrix.h"


namespace tbml {

	class NeuralNetwork {

	protected:
		size_t layerCount;
		std::vector<size_t> layerSizes;
		std::vector<Matrix> weights;
		std::vector<Matrix> bias;
		float (*activator)(float);
		std::vector<Matrix> neuronOutCache;

	public:
		NeuralNetwork(std::vector<size_t> layerSizes_);
		NeuralNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float));

		Matrix propogate(Matrix input);
		void printLayers();

		std::vector<Matrix>& getWeights();
		std::vector<Matrix>& getBias();
	};
}
