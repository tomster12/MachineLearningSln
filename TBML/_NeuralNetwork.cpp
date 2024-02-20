#include "stdafx.h"
#include "_NeuralNetwork.h"
#include "_Utility.h"
#include "omp.h"

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

			// Propogate first layer, allow for column vector input
			if (input.getShape().size() == 1)
				layers[0]->propogate(input.transposed());
			else layers[0]->propogate(input);

			// Propogate rest of layers, using const refs
			for (size_t i = 1; i < layers.size(); i++)
			{
				layers[i]->propogate(layers[i - 1]->getOutput());
			}

			// Return predicted of last layer
			return layers[layers.size() - 1]->getOutput();
		}

		void _NeuralNetwork::train(const _Tensor& input, const _Tensor& expected, const _TrainingConfig& config)
		{
			// Batch data if needed
			size_t batchCount;
			std::vector<_Tensor> inputBatches, expectedBatches;
			if (config.batchSize == -1)
			{
				// Do not batch data
				inputBatches = std::vector<_Tensor>({ input });
				expectedBatches = std::vector<_Tensor>({ expected });
				batchCount = 1;
			}
			else
			{
				// Split into batches
				inputBatches = input.groupRows(config.batchSize);
				expectedBatches = expected.groupRows(config.batchSize);
				assert(inputBatches.size() == expectedBatches.size() && "Input and expected batch count mismatch");
				batchCount = inputBatches.size();
			}

			std::chrono::steady_clock::time_point tTrainStart = std::chrono::steady_clock::now();
			std::chrono::steady_clock::time_point tEpochStart = tTrainStart;
			std::chrono::steady_clock::time_point tBatchStart = tTrainStart;

			// Train until max epochs or error threshold
			int epoch = 0;
			int maxEpochs = (config.epochs == -1 && config.errorThreshold > 0.0f) ? MAX_EPOCHS : config.epochs;
			if (config.logLevel > 0) printf("Training started for %d epochs\n", maxEpochs);
			for (; epoch < maxEpochs; epoch++)
			{
				float epochLoss = 0.0f;

				// Train each batch
				for (size_t batch = 0; batch < batchCount; batch++)
				{
					// Propogate input then calculate loss
					const _Tensor& predicted = propogate(inputBatches[batch]);
					float batchLoss = lossFn.activate(predicted, expectedBatches[batch]);
					epochLoss += batchLoss / batchCount;

					// Backpropogate loss and then each layer
					_Tensor pdLossToOut = lossFn.derive(predicted, expectedBatches[batch]);
					layers[layers.size() - 1]->backpropogate(pdLossToOut);
					for (int i = (int)layers.size() - 2; i >= 0; i--)
					{
						layers[i]->backpropogate(layers[i + 1]->getPdToIn());
					}

					// Apply gradient descent
					for (size_t j = 0; j < layers.size(); j++)
					{
						layers[j]->gradientDescent(config.learningRate, config.momentumRate);
					}

					if (config.logLevel == 3)
					{
						std::chrono::steady_clock::time_point tBatchEnd = std::chrono::steady_clock::now();
						float accuracy = fn::_classificationAccuracy(predicted, expectedBatches[batch]);
						auto us = std::chrono::duration_cast<std::chrono::microseconds>(tBatchEnd - tBatchStart);
						printf("Epoch %d, Batch %d: Loss: %f, Accuracy: %f\%, Time: %fms\n", epoch, (int)batch, batchLoss, accuracy * 100, us.count() / 1000.0f);
						tBatchStart = tBatchEnd;
					}
				}

				if (config.logLevel == 2)
				{
					std::chrono::steady_clock::time_point tEpochEnd = std::chrono::steady_clock::now();
					_Tensor predicted = propogate(input);
					float accuracy = fn::_classificationAccuracy(predicted, expected);
					auto us = std::chrono::duration_cast<std::chrono::microseconds>(tEpochEnd - tEpochStart);
					printf("Epoch %d: Loss: %f, Accuracy: %f, Time: %fms\n", epoch, epochLoss, accuracy, us.count() / 1000.0f);
					tEpochStart = tEpochEnd;
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
			assert(input.getDims() == 2 && input.getShape(1) == weights.getShape(0) && "Input shape does not match weights shape");

			// Propogate input with weights and bias
			propogateInput = &input;
			output = input.matmulled(weights).add(bias, 0);
			activationFn.activate(output);
		}

		void _DenseLayer::backpropogate(const _Tensor& pdToOut)
		{
			assert(pdToOut.getDims() == 2 && pdToOut.getShape(1) == weights.getShape(1) && "pdToOut shape does not match weights shape");

			// Calculate pd to neuron in and layer in
			_Tensor pdToNet = activationFn.chainDerivative(output, pdToOut);
			pdToIn = pdToNet.matmulled(weights.transposed());

			// Setup variables for derivatives
			int batchSize = (int)propogateInput->getShape(0);
			int m = (int)weights.getShape(0);
			int n = (int)weights.getShape(1);
			pdToWeights = _Tensor(weights.getShape(), 0);
			pdToBias = _Tensor(bias.getShape(), 0);

			// Calculate pd to weights and bias as average of batch
			#pragma omp parallel for num_threads(4)
			for (int batchRow = 0; batchRow < batchSize; batchRow++)
			{
				for (int i = 0; i < m; i++)
				{
					for (int j = 0; j < n; j++)
					{
						pdToWeights(i, j) += ((*propogateInput)(batchRow, i) * pdToNet(batchRow, j)) / batchSize;
					}
				}

				for (int j = 0; j < n; j++)
				{
					pdToBias(0, j) += pdToNet(batchRow, j) / batchSize;
				}
			}
		}

		void _DenseLayer::gradientDescent(float learningRate, float momentumRate)
		{
			// Apply gradient descent with momentum
			momentumWeights = (momentumWeights * momentumRate) - (pdToWeights * learningRate);
			momentumBias = (momentumBias * momentumRate) - (pdToBias * learningRate);
			weights += momentumWeights;
			bias += momentumBias;
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
			output = _Tensor::ZERO;
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
			output = _Tensor::ZERO;
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
			output = _Tensor::ZERO;
		}

		void _MaxPoolLayer::backpropogate(const _Tensor& pdToOut)
		{
			// TODO: Implement
			pdToIn = pdToOut;
		}
	}
}
