#include "stdafx.h"
#include "_NeuralNetwork.h"
#include "_Utility.h"

namespace tbml
{
	namespace nn
	{
		_NeuralNetwork::_NeuralNetwork(fn::_LossFunction&& lossFn)
			: lossFn(lossFn)
		{}

		_NeuralNetwork::_NeuralNetwork(fn::_LossFunction&& lossFn, std::vector< _Layer*> layers)
			: lossFn(lossFn), layers(layers)
		{}

		_NeuralNetwork::~_NeuralNetwork()
		{
			for (size_t i = 0; i < layers.size(); i++) delete layers[i];
			layers.clear();
		}

		void _NeuralNetwork::addLayer(_Layer* layer)
		{
			layers.push_back(layer);
		}

		const _Tensor& _NeuralNetwork::propogate(const _Tensor& input)
		{
			if (layers.size() == 0) return _Tensor::ZERO;

			// Allow for column vector input
			if (input.getShape().size() == 1)
				layers[0]->propogate(input.transposed());
			else layers[0]->propogate(input);

			// Propogate layers using const refs
			for (size_t i = 1; i < layers.size(); i++)
			{
				layers[i]->propogate(layers[i - 1]->getPredicted());
			}

			return layers[layers.size() - 1]->getPredicted();
		}

		void _NeuralNetwork::train(const _Tensor& input, const _Tensor& expected, const _TrainingConfig& config)
		{
			int epoch = 0;
			int maxEpochs = (config.epochs == -1 && config.errorThreshold > 0.0f) ? MAX_EPOCHS : config.epochs;

			std::chrono::steady_clock::time_point tTrainStart = std::chrono::steady_clock::now();
			std::chrono::steady_clock::time_point tEpochStart = tTrainStart;

			if (config.logLevel > 0) printf("Training started for %d epochs\n", maxEpochs);

			// Train for maxEpochs or until errorThreshold is reached
			for (; epoch < maxEpochs; epoch++)
			{
				// Propogate input and calculate loss
				const _Tensor& output = propogate(input);
				float epochLoss = lossFn.activate(output, expected);

				if (config.logLevel == 2)
				{
					std::chrono::steady_clock::time_point tEpochEnd = std::chrono::steady_clock::now();
					float accuracy = fn::_classificationAccuracy(output, expected);
					auto us = std::chrono::duration_cast<std::chrono::microseconds>(tEpochEnd - tEpochStart);
					printf("Epoch %d: Loss: %f, Accuracy: %f, Time: %fms\n", epoch, epochLoss, accuracy, us.count() / 1000.0f);
					tEpochStart = tEpochEnd;
				}

				if (epochLoss < config.errorThreshold) break;

				// Backpropogate and apply gradient descent
				backpropogate(output, expected);
				for (size_t j = 0; j < layers.size(); j++)
				{
					layers[j]->gradientDescent(config.learningRate, config.momentumRate);
				}
			}

			if (config.logLevel >= 1)
			{
				std::chrono::steady_clock::time_point tTrainEnd = std::chrono::steady_clock::now();
				auto us = std::chrono::duration_cast<std::chrono::microseconds>(tTrainEnd - tTrainStart);
				printf("Training complete for %d epochs, Time taken: %fms\n\n", epoch, us.count() / 1000.0f);
			}
		}

		void _NeuralNetwork::print() const
		{
			for (const auto& layer : layers) layer->print();
		}

		void _NeuralNetwork::backpropogate(const _Tensor& predicted, const _Tensor& expected)
		{
			if (layers.size() == 0) return;

			// Backpropogate loss function
			_Tensor pdLossToOut = lossFn.derive(predicted, expected);

			// Backpropogate layers using next layers pd to in
			layers[layers.size() - 1]->backpropogate(pdLossToOut);
			for (int i = (int)layers.size() - 2; i >= 0; i--)
			{
				layers[i]->backpropogate(layers[i + 1]->getPdToIn());
			}
		}

		_DenseLayer::_DenseLayer(size_t inputSize, size_t outputSize, fn::_ActivationFunction&& activationFn, _DenseInitType initType, bool useBias)
			: activationFn(activationFn)
		{
			weights = _Tensor({ inputSize, outputSize }, 0);

			if (initType == _DenseInitType::RANDOM)
			{
				weights.map([](float _) { return fn::getRandomFloat() * 2 - 1; });
			}

			if (useBias)
			{
				bias = _Tensor({ 1, outputSize }, 0);

				if (initType == _DenseInitType::RANDOM)
				{
					bias.map([](float _) { return fn::getRandomFloat() * 2 - 1; });
				}
			}
		}

		void _DenseLayer::propogate(const _Tensor& input)
		{
			propogateInput = RefHolder<_Tensor>(input);
			predicted = input.matmulled(weights).add(bias, 0);
			activationFn.activate(predicted);
		}

		void _DenseLayer::backpropogate(const _Tensor& pdToOut)
		{
			// Calculate pd to neuron in and layer in
			_Tensor pdToNet = activationFn.chainDerivative(predicted, pdToOut);
			pdToIn = pdToNet.matmulled(weights.transposed());

			// Calculate pd to weights and bias
			GDPdErrorToWeights = propogateInput().matmulled(pdToNet);
			GDPdErrorToBias = pdToNet;
		}

		void _DenseLayer::gradientDescent(float learningRate, float momentumRate)
		{
			// Apply gradient descent with momentum
			GDMomentumWeights = (GDMomentumWeights * momentumRate) + (GDPdErrorToWeights * -learningRate);
			GDMomentumBias = (GDMomentumBias * momentumRate) + (GDPdErrorToBias * -learningRate);
			weights += GDMomentumWeights;
			bias += GDMomentumBias;
		}

		void _DenseLayer::print() const
		{
			weights.print("Weights:");
			bias.print("Bias:");
		}

		_ConvLayer::_ConvLayer(std::vector<size_t> kernel, std::vector<size_t> stride, fn::_ActivationFunction&& activationFn)
			: activationFn(activationFn)
		{}

		void _ConvLayer::propogate(const _Tensor& input)
		{
			// TODO: Implement
			predicted = _Tensor::ZERO;
		}

		void _ConvLayer::backpropogate(const _Tensor& pdToOut)
		{
			// TODO: Implement
			pdToIn = pdToOut;
		}

		void _ConvLayer::gradientDescent(float learningRate, float momentumRate)
		{
			// TODO: Implement
		}

		_FlattenLayer::_FlattenLayer()
		{}

		void _FlattenLayer::propogate(const _Tensor& input)
		{
			// TODO: Implement
			predicted = _Tensor::ZERO;
		}

		void _FlattenLayer::backpropogate(const _Tensor& pdToOut)
		{
			// TODO: Implement
			pdToIn = pdToOut;
		}

		_MaxPoolLayer::_MaxPoolLayer()
		{}

		void _MaxPoolLayer::propogate(const _Tensor& input)
		{
			// TODO: Implement
			predicted = _Tensor::ZERO;
		}

		void _MaxPoolLayer::backpropogate(const _Tensor& pdToOut)
		{
			// TODO: Implement
			pdToIn = pdToOut;
		}
	}
}
