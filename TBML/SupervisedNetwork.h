
#pragma once

#include "Matrix.h"
#include "NeuralNetwork.h"


namespace tbml
{
	struct TrainingConfig { int epochs = 20; int batchSize = -1; float learningRate = 0.1f; float momentumRate = 0.1f; float errorExit = 0.0f; int logLevel = 0; };


	class SupervisedNetwork : public NeuralNetwork
	{
	private:
		static const int MAX_maxIterations = 1'000'000;
		Matrix pdOutCache;
		std::vector<Matrix> pdNeuronInCache;
		std::vector<std::vector<Matrix>> pdWeightsCache;
		std::vector<std::vector<Matrix>> pdBiasCache;
		std::vector<Matrix> pdWeightsMomentum;
		std::vector<Matrix> pdBiasMomentum;
		float (*activatorPd)(float);
		float (*calcError)(Matrix, Matrix);
		Matrix(*calcErrorPd)(Matrix, Matrix);
		
		void calculateDerivates(Matrix& predicted, Matrix& expected);
		void calculatePdErrorToIn(size_t layer);
		Matrix pdErrorToOut(size_t layer);

	public:
		SupervisedNetwork(std::vector<size_t> layerSizes_);
		SupervisedNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float));
		SupervisedNetwork(std::vector<size_t> layerSizes_, float (*calcError)(Matrix, Matrix), Matrix(*calcErrorPd)(Matrix, Matrix));
		SupervisedNetwork(std::vector<size_t> layerSizes_, float (*activator_)(float), float (*activatorPd_)(float), float (*calcError_)(Matrix, Matrix), Matrix(*calcErrorPd_)(Matrix, Matrix));

		void train(Matrix input, Matrix expected, TrainingConfig config);
	};
}
