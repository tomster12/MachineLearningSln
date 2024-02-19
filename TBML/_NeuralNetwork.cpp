#include "stdafx.h"
#include "_NeuralNetwork.h"

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

			// Propogate layers using const refs
			layers[0]->propogate(input);
			for (size_t i = 1; i < layers.size(); i++)
			{
				layers[i]->propogate(layers[i - 1]->getPropogateOutput());
			}
			return layers[layers.size() - 1]->getPropogateOutput();
		}

		void _NeuralNetwork::train(const std::vector<_Tensor>& inputs, const std::vector<_Tensor>& expectedOutputs, const _TrainingConfig& config)
		{
			// Stochastic gradient descent without batches
			for (size_t i = 0; i < inputs.size(); i++)
			{
				// Propogate input to output
				const _Tensor& output = propogate(inputs[i]);

				// Backpropogate output through network
				backpropogate(output, expectedOutputs[i]);

				// Update weights and biases
				for (size_t j = 0; j < layers.size(); j++)
				{
					// TODO: Implement
				}
			}
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
				layers[i]->backpropogate(layers[i + 1]->getBackpropogateOutput());
			}
		}

		_DenseLayer::_DenseLayer(size_t inputSize, size_t outputSize, fn::_ActivationFunction&& actFn, _DenseInitType initType, bool useBias)
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
			// TODO: Fix with bias being wrong
			propogateOutput = input.transposed().matmul(weights);
		}

		void _DenseLayer::backpropogate(const _Tensor& pdToOut)
		{
			// TODO: Implement
			backpropogateOutput = pdToOut;
		}

		_ConvLayer::_ConvLayer(std::vector<size_t> kernel, std::vector<size_t> stride, fn::_ActivationFunction&& actFn)
		{}

		void _ConvLayer::propogate(const _Tensor& input)
		{
			// TODO: Implement
			propogateOutput = _Tensor::ZERO;
		}

		void _ConvLayer::backpropogate(const _Tensor& pdToOut)
		{
			// TODO: Implement
			backpropogateOutput = pdToOut;
		}

		_FlattenLayer::_FlattenLayer()
		{}

		void _FlattenLayer::propogate(const _Tensor& input)
		{
			// TODO: Implement
			propogateOutput = _Tensor::ZERO;
		}

		void _FlattenLayer::backpropogate(const _Tensor& pdToOut)
		{
			// TODO: Implement
			backpropogateOutput = pdToOut;
		}

		_MaxPoolLayer::_MaxPoolLayer()
		{}

		void _MaxPoolLayer::propogate(const _Tensor& input)
		{
			// TODO: Implement
			propogateOutput = _Tensor::ZERO;
		}

		void _MaxPoolLayer::backpropogate(const _Tensor& pdToOut)
		{
			// TODO: Implement
			backpropogateOutput = pdToOut;
		}
	}
}
