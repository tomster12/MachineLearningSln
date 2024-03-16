#include "stdafx.h"
#include "NeuralNetwork.h"
#include "Utility.h"
#include "omp.h"

namespace tbml
{
	namespace nn
	{
		void NeuralNetwork::addLayer(LayerPtr&& layer)
		{
			layers.push_back(std::move(layer));
		}

		const Tensor& NeuralNetwork::propogate(const Tensor& input)
		{
			if (layers.size() == 0) return Tensor::ZERO;

			// Propogate first layer
			layers[0]->propogate(input);

			// Propogate rest of layers, using const refs
			for (size_t i = 1; i < layers.size(); i++)
			{
				layers[i]->propogate(layers[i - 1]->getOutput());
			}

			// Return predicted output of last layer
			return layers[layers.size() - 1]->getOutput();
		}

		Tensor NeuralNetwork::propogate(const Tensor& input) const
		{
			if (layers.size() == 0) return Tensor(Tensor::ZERO);

			// Copy input into a local tensor
			Tensor current = input;

			// Propogate each layer without copies
			layers[0]->propogateMut(current);
			for (size_t i = 0; i < layers.size(); i++)
			{
				current = layers[i]->propogate(current);
			}

			// Return predicted of last layer
			return std::move(current);
		}

		void NeuralNetwork::train(const Tensor& input, const Tensor& expected, const TrainingConfig& config)
		{
			// Batch data if needed
			size_t batchCount;
			std::vector<Tensor> inputBatches, expectedBatches;
			if (config.batchSize == -1)
			{
				// Do not batch data
				inputBatches = std::vector<Tensor>({ input });
				expectedBatches = std::vector<Tensor>({ expected });
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
			int maxEpochs = config.epochs == -1 ? MAX_EPOCHS : config.epochs;
			if (config.logLevel > 0) printf("Training started for %d epochs\n", maxEpochs);
			for (; epoch < maxEpochs; epoch++)
			{
				float epochLoss = 0.0f;

				// Train each batch
				for (size_t batch = 0; batch < batchCount; batch++)
				{
					// Propogate input then calculate loss
					const Tensor& predicted = propogate(inputBatches[batch]);
					float batchLoss = lossFn->activate(predicted, expectedBatches[batch]);
					epochLoss += batchLoss / batchCount;

					// Backpropogate loss and then each layer
					Tensor pdLossToOut = lossFn->derive(predicted, expectedBatches[batch]);
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

					if (config.logLevel >= 3)
					{
						std::chrono::steady_clock::time_point tBatchEnd = std::chrono::steady_clock::now();
						float accuracy = fn::classificationAccuracy(predicted, expectedBatches[batch]);
						auto us = std::chrono::duration_cast<std::chrono::microseconds>(tBatchEnd - tBatchStart);
						printf("Epoch %d, Batch %d: Loss: %f, Accuracy: %f, Time: %fms\n", epoch, (int)batch, batchLoss, accuracy * 100, us.count() / 1000.0f);
						tBatchStart = tBatchEnd;
					}
				}

				if (config.logLevel >= 2)
				{
					std::chrono::steady_clock::time_point tEpochEnd = std::chrono::steady_clock::now();
					Tensor predicted = propogate(input);
					float accuracy = fn::classificationAccuracy(predicted, expected);
					auto us = std::chrono::duration_cast<std::chrono::microseconds>(tEpochEnd - tEpochStart);
					printf("Epoch %d: Loss: %f, Accuracy: %f, Time: %fms\n", epoch, epochLoss, accuracy, us.count() / 1000.0f);
					tEpochStart = tEpochEnd;
				}

				// Exit if error threshold is met
				if (epochLoss < config.errorThreshold) break;
			}

			if (config.logLevel >= 1)
			{
				std::chrono::steady_clock::time_point tTrainEnd = std::chrono::steady_clock::now();
				auto us = std::chrono::duration_cast<std::chrono::microseconds>(tTrainEnd - tTrainStart);
				printf("Training complete for %d epochs, Time taken: %fms\n\n", epoch, us.count() / 1000.0f);
			}
		}

		void NeuralNetwork::print() const
		{
			for (const auto& layer : layers) layer->print();
		}

		void NeuralNetwork::saveToFile(const std::string& filename) const
		{
			std::ofstream file(filename, std::ios::binary);
			if (!file.is_open())
			{
				throw std::runtime_error("Failed to open file for writing");
			}

			// Write loss function
			lossFn->serialize(file);

			// Write number of layers
			file << layers.size() << "\n";

			// Write each layer
			for (const auto& layer : layers)
			{
				layer->serialize(file);
			}
		}

		NeuralNetwork NeuralNetwork::loadFromFile(const std::string& filename)
		{
			std::ifstream file(filename, std::ios::binary);
			if (!file.is_open())
			{
				throw std::runtime_error("Failed to open file for reading");
			}

			// Read the loss function
			fn::LossFunctionPtr lossFn = fn::LossFunction::deserialize(file);

			// Read the number of layers
			size_t layerCount;
			file >> layerCount;

			// Read each layer
			std::vector<LayerPtr> layers;
			for (size_t i = 0; i < layerCount; i++)
			{
				layers.push_back(Layer::deserialize(file));
			}

			return NeuralNetwork(std::move(lossFn), std::move(layers));
		}

		std::shared_ptr<Layer> Layer::deserialize(std::istream& is)
		{
			std::string type;
			is >> type;

			if (type == "Dense")
			{
				fn::ActivationFunctionPtr activationFn = fn::ActivationFunction::deserialize(is);
				Tensor weights = Tensor::deserialize(is);
				Tensor bias = Tensor::deserialize(is);
				return std::make_shared<DenseLayer>(std::move(weights), std::move(bias), std::move(activationFn));
			}

			throw std::runtime_error("Unknown layer type");
		}

		DenseLayer::DenseLayer(const DenseLayer& other)
		{
			weights = other.weights;
			bias = other.bias;
			activationFn = fn::ActivationFunctionPtr(other.activationFn);
		}

		DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, fn::ActivationFunctionPtr&& activationFn, _DenseInitType initType, bool useBias)
			: activationFn(activationFn)
		{
			weights = Tensor({ inputSize, outputSize }, 0);

			if (initType == _DenseInitType::RANDOM)
			{
				weights.map([](float _) { return fn::getRandomFloat() * 2 - 1; });
			}

			if (useBias)
			{
				bias = Tensor({ 1, outputSize }, 0);

				if (initType == _DenseInitType::RANDOM)
				{
					bias.map([](float _) { return fn::getRandomFloat() * 2 - 1; });
				}
			}
		}

		DenseLayer::DenseLayer(Tensor&& weights, Tensor&& bias, fn::ActivationFunctionPtr&& activationFn)
			: weights(std::move(weights)), bias(std::move(bias)), activationFn(std::move(activationFn))
		{}

		const Tensor& DenseLayer::propogate(const Tensor& input)
		{
			assert(input.getDims() == 2 && input.getShape(1) == weights.getShape(0) && "Input shape does not match weights shape");

			// Propogate input with weights and bias
			propogateInput = &input;
			output = input.matmulled(weights).add(bias, 0);
			activationFn->activate(output);
			return output;
		}

		Tensor DenseLayer::propogate(const Tensor& input) const
		{
			assert(input.getDims() == 2 && input.getShape(1) == weights.getShape(0) && "Input shape does not match weights shape");

			Tensor output = input.matmulled(weights).add(bias, 0);
			activationFn->activate(output);
			return output;
		}

		void DenseLayer::propogateMut(Tensor& input) const
		{
			assert(input.getDims() == 2 && input.getShape(1) == weights.getShape(0) && "Input shape does not match weights shape");

			input.matmul(weights).add(bias, 0);
			activationFn->activate(input);
		}

		void DenseLayer::backpropogate(const Tensor& pdToOut)
		{
			assert(pdToOut.getDims() == 2 && pdToOut.getShape(1) == weights.getShape(1) && "pdToOut shape does not match weights shape");

			// Calculate pd to neuron in and layer in
			Tensor pdToNet = activationFn->chainDerivative(output, pdToOut);
			pdToIn = pdToNet.matmulled(weights.transposed());

			// Setup variables for derivatives
			int batchSize = (int)propogateInput->getShape(0);
			int m = (int)weights.getShape(0);
			int n = (int)weights.getShape(1);
			pdToWeights = Tensor(weights.getShape(), 0);
			pdToBias = Tensor(bias.getShape(), 0);

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

		void DenseLayer::gradientDescent(float learningRate, float momentumRate)
		{
			// Apply gradient descent with momentum
			momentumWeights = (momentumWeights * momentumRate) - (pdToWeights * learningRate);
			momentumBias = (momentumBias * momentumRate) - (pdToBias * learningRate);
			weights += momentumWeights;
			bias += momentumBias;
		}

		void DenseLayer::print() const
		{
			weights.print("Weights:");
			bias.print("Bias:");
		}

		LayerPtr DenseLayer::clone() const
		{
			return std::make_shared<DenseLayer>(*this);
		}

		void DenseLayer::serialize(std::ostream& os) const
		{
			os << "Dense\n";
			activationFn->serialize(os);
			weights.serialize(os);
			bias.serialize(os);
		}

		/*
		ConvLayer::ConvLayer(std::vector<size_t> kernel, std::vector<size_t> stride, fn::ActivationFunction&& activationFn)
			: activationFn(activationFn)
		{}

		const Tensor& ConvLayer::propogate(const Tensor& input)
		{
			// TODO: Implement
			output = Tensor::ZERO;
			return output;
		}

		Tensor ConvLayer::propogate(const Tensor& input) const
		{
			// TODO: Implement
			return Tensor::ZERO;
		}

		void ConvLayer::backpropogate(const Tensor& pdToOut)
		{
			// TODO: Implement
			pdToIn = pdToOut;
		}

		void ConvLayer::gradientDescent(float learningRate, float momentumRate)
		{
			// TODO: Implement
		}

		LayerPtr ConvLayer::clone() const
		{
			// TODO: Implement
			return std::shared_ptr<ConvLayer>();
		}

		FlattenLayer::FlattenLayer()
		{}

		const Tensor& FlattenLayer::propogate(const Tensor& input)
		{
			// TODO: Implement
			output = Tensor::ZERO;
			return output;
		}

		Tensor FlattenLayer::propogate(const Tensor& input) const
		{
			// TODO: Implement
			return Tensor::ZERO;
		}

		void FlattenLayer::backpropogate(const Tensor& pdToOut)
		{
			// TODO: Implement
			pdToIn = pdToOut;
		}

		LayerPtr FlattenLayer::clone() const
		{
			// TODO: Implement
			return std::shared_ptr<FlattenLayer>();
		}

		MaxPoolLayer::MaxPoolLayer()
		{}

		const Tensor& MaxPoolLayer::propogate(const Tensor& input)
		{
			// TODO: Implement
			output = Tensor::ZERO;
			return output;
		}

		Tensor MaxPoolLayer::propogate(const Tensor& input) const
		{
			// TODO: Implement
			return Tensor::ZERO;
		}

		void MaxPoolLayer::backpropogate(const Tensor& pdToOut)
		{
			// TODO: Implement
			pdToIn = pdToOut;
		}

		LayerPtr MaxPoolLayer::clone() const
		{
			// TODO: Implement
			return std::shared_ptr<MaxPoolLayer>();
		}
		*/
	}
}
