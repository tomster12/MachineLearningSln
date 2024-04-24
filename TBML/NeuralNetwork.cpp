#include "stdafx.h"
#include "NeuralNetwork.h"
#include "Utility.h"
#include "omp.h"

namespace tbml
{
	namespace nn
	{
		namespace Layer
		{
			BasePtr deserialize(std::istream& is)
			{
				std::string type;
				is >> type;

				if (type == "Dense")
				{
					fn::ActivationFunctionPtr activationFn = fn::ActivationFunction::deserialize(is);
					Tensor weights = Tensor::deserialize(is);
					Tensor bias = Tensor::deserialize(is);
					return std::make_shared<Dense>(std::move(weights), std::move(bias), std::move(activationFn));
				}

				throw std::runtime_error("Unknown layer type");
			}

			Dense::Dense(const Dense& other)
			{
				weights = other.weights;
				bias = other.bias;
				activationFn = fn::ActivationFunctionPtr(other.activationFn);
			}

			Dense::Dense(size_t inputSize, size_t outputSize, fn::ActivationFunctionPtr&& activationFn, DenseInitType initType, bool useBias)
				: activationFn(activationFn)
			{
				weights = Tensor({ inputSize, outputSize }, 0);

				if (initType == DenseInitType::RANDOM)
				{
					weights.map([](float _) { return fn::getRandomFloat() * 2 - 1; });
				}

				if (useBias)
				{
					bias = Tensor({ 1, outputSize }, 0);

					if (initType == DenseInitType::RANDOM)
					{
						bias.map([](float _) { return fn::getRandomFloat() * 2 - 1; });
					}
				}
			}

			Dense::Dense(Tensor&& weights, Tensor&& bias, fn::ActivationFunctionPtr&& activationFn)
				: weights(std::move(weights)), bias(std::move(bias)), activationFn(std::move(activationFn))
			{}

			void Dense::propogateMut(Tensor& input) const
			{
				assert(input.getDims() == 2 && input.getShape(1) == weights.getShape(0) && "Input shape does not match weights shape");

				// Mutably propogate input with weights and bias
				input.matmul(weights).add(bias, 0);
				activationFn->activate(input);
			}

			const Tensor& Dense::propogateRef(const Tensor& input)
			{
				assert(input.getDims() == 2 && input.getShape(1) == weights.getShape(0) && "Input shape does not match weights shape");

				// Propogate input with weights and bias
				// Retain input and output for backprop
				propogateInput = &input;
				output = input.matmulled(weights).add(bias, 0);
				activationFn->activate(output);
				return output;
			}

			void Dense::backpropogate(const Tensor& gradOutput)
			{
				assert(gradOutput.getDims() == 2 && gradOutput.getShape(1) == weights.getShape(1) && "gradOutput shape does not match weights shape");
				assert(this->retainValues && "Cannot backpropogate when not retaining values");

				// Calculate pd to neuron in and layer in
				Tensor gradNet = activationFn->chainDerivative(output, gradOutput);
				gradInput = gradNet.matmulled(weights.transposed());

				// Calculate pd to weights and bias as average of batches
				int batchSize = (int)propogateInput->getShape(0);
				int m = (int)weights.getShape(0);
				int n = (int)weights.getShape(1);
				gradWeights = Tensor(weights.getShape(), 0);
				gradBias = Tensor(bias.getShape(), 0);

				#pragma omp parallel for num_threads(12)
				for (int batchRow = 0; batchRow < batchSize; batchRow++)
				{
					for (int i = 0; i < m; i++)
					{
						for (int j = 0; j < n; j++)
						{
							gradWeights(i, j) += ((*propogateInput)(batchRow, i) * gradNet(batchRow, j)) / batchSize;
						}
					}

					for (int j = 0; j < n; j++)
					{
						gradBias(0, j) += gradNet(batchRow, j) / batchSize;
					}
				}
			}

			void Dense::gradientDescent(float learningRate, float momentumRate)
			{
				// Apply gradient descent with momentum
				momentumWeights = (momentumWeights * momentumRate) - (gradWeights * learningRate);
				momentumBias = (momentumBias * momentumRate) - (gradBias * learningRate);
				weights += momentumWeights;
				bias += momentumBias;
			}

			void Dense::print() const
			{
				weights.print("Weights:");
				bias.print("Bias:");
			}

			BasePtr Dense::clone() const
			{
				return std::make_shared<Dense>(*this);
			}

			void Dense::serialize(std::ostream& os) const
			{
				os << "Dense\n";
				activationFn->serialize(os);
				weights.serialize(os);
				bias.serialize(os);
			}
		}

		void NeuralNetwork::addLayer(Layer::BasePtr&& layer)
		{
			layers.push_back(std::move(layer));
		}

		Tensor NeuralNetwork::propogate(const Tensor& input) const
		{
			if (layers.size() == 0) return Tensor(Tensor::ZERO);

			// Copy to local, propogate layers mutably
			Tensor current = input;
			for (size_t i = 0; i < layers.size(); i++) layers[i]->propogateMut(current);
			return current;
		}

		void NeuralNetwork::propogateMut(Tensor& input) const
		{
			if (layers.size() == 0) return;

			// Propogate layers with mutable input
			for (size_t i = 0; i < layers.size(); i++) layers[i]->propogateMut(input);
		}

		const Tensor& NeuralNetwork::propogateRef(const Tensor& input)
		{
			if (layers.size() == 0) return Tensor::ZERO;

			// Copy to local, propogate layers mutably, using references
			// Used to retain values for backpropogation
			layers[0]->propogateRef(input);
			for (size_t i = 1; i < layers.size(); i++) layers[i]->propogateRef(layers[i - 1]->getOutput());
			return layers[layers.size() - 1]->getOutput();
		}

		void NeuralNetwork::train(const Tensor& input, const Tensor& expected, const TrainingConfig& config)
		{
			// Batch data if needed
			// TODO: Batch randomly each epoch
			size_t batchCount;
			std::vector<Tensor> inputBatches, expectedBatches;
			if (config.batchSize <= 1)
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

			// Set layers to retain values
			for (const auto& layer : layers) layer->setRetainValues(true);

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
					const Tensor& predicted = propogateRef(inputBatches[batch]);
					float batchLoss = lossFn->calculate(predicted, expectedBatches[batch]);
					epochLoss += batchLoss / batchCount;

					// Backpropogate loss and then each layer
					Tensor pdLossToOut = lossFn->derivative(predicted, expectedBatches[batch]);
					layers[layers.size() - 1]->backpropogate(pdLossToOut);
					for (int i = (int)layers.size() - 2; i >= 0; i--)
					{
						layers[i]->backpropogate(layers[i + 1]->getGradInput());
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
						printf("Epoch %d, Batch %d: Loss: %.3f, Accuracy: %.1f%%, Time: %.3fms\n", epoch, (int)batch, batchLoss, accuracy * 100, us.count() / 1000.0f);
						tBatchStart = tBatchEnd;
					}
				}

				if (config.logLevel >= 2)
				{
					std::chrono::steady_clock::time_point tEpochEnd = std::chrono::steady_clock::now();
					const Tensor& predicted = propogateRef(input);
					float accuracy = fn::classificationAccuracy(predicted, expected);
					auto us = std::chrono::duration_cast<std::chrono::microseconds>(tEpochEnd - tEpochStart);
					printf("Epoch %d: Loss: %.3f, Accuracy: %.1f%%, Time: %.3fms\n", epoch, epochLoss, accuracy * 100, us.count() / 1000.0f);
					tEpochStart = tEpochEnd;
				}

				// Exit if error threshold is met
				if (epochLoss < config.errorThreshold) break;
			}

			if (config.logLevel >= 1)
			{
				std::chrono::steady_clock::time_point tTrainEnd = std::chrono::steady_clock::now();
				auto us = std::chrono::duration_cast<std::chrono::microseconds>(tTrainEnd - tTrainStart);
				printf("Training complete for %d epochs, Time taken: %.3fms\n\n", epoch, us.count() / 1000.0f);
			}

			// Reset layers to not retain values
			for (const auto& layer : layers) layer->setRetainValues(false);
		}

		int NeuralNetwork::getParameterCount() const
		{
			int count = 0;
			for (const auto& layer : layers) count += layer->getParameterCount();
			return count;
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
			std::vector<Layer::BasePtr> layers;
			for (size_t i = 0; i < layerCount; i++)
			{
				layers.push_back(Layer::deserialize(file));
			}

			return NeuralNetwork(std::move(lossFn), std::move(layers));
		}
	}
}
