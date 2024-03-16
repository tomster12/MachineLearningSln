#pragma once

#include "Utility.h"
#include "Tensor.h"

namespace tbml
{
	namespace nn
	{
		struct TrainingConfig
		{
			int epochs = 20;
			int batchSize = -1;
			float learningRate = 0.1f;
			float momentumRate = 0.1f;
			float errorThreshold = 0.0f;
			size_t logLevel = 0;
		};

		class Layer
		{
		public:
			virtual const Tensor& propogate(const Tensor& input) = 0;
			virtual Tensor propogate(const Tensor& input) const = 0;
			virtual void backpropogate(const Tensor& pdToOut) = 0;
			virtual void gradientDescent(float learningRate, float momentumRate) {};
			virtual std::vector<size_t> getInputShape() const { return {}; }; // TODO: Make pure virtual
			virtual std::vector<size_t> getOutputShape() const { return {}; }; // TODO: Make pure virtual
			const Tensor& getOutput() const { return output; };
			const Tensor& getPdToIn() const { return pdToIn; };
			virtual void print() const {};
			virtual std::shared_ptr<Layer> clone() const = 0;

		protected:
			Tensor output;
			Tensor pdToIn;
		};

		class NeuralNetwork
		{
		public:
			static const int MAX_EPOCHS = 1'000;

			NeuralNetwork() {}
			NeuralNetwork(fn::LossFunction&& lossFn) : lossFn(std::move(lossFn)) {}
			NeuralNetwork(fn::LossFunction&& lossFn, std::vector<std::shared_ptr<Layer>>&& layers) : lossFn(std::move(lossFn)), layers(std::move(layers)) {}
			void addLayer(std::shared_ptr<Layer>&& layer);
			const Tensor& propogate(const Tensor& input);
			Tensor propogate(const Tensor& input) const;
			void train(const Tensor& input, const Tensor& expected, const TrainingConfig& config);
			std::vector<size_t> getInputShape() const { return layers[0]->getInputShape(); }
			std::vector<size_t> getOutputShape() const { return layers[layers.size() - 1]->getOutputShape(); }
			const std::vector<std::shared_ptr<Layer>>& getLayers() const { return layers; }
			fn::LossFunction getLossFunction() const { return lossFn; }
			void print() const;

		private:
			fn::LossFunction lossFn;
			std::vector<std::shared_ptr<Layer>> layers;
		};

		enum class _DenseInitType { ZERO, RANDOM };

		class DenseLayer : public Layer
		{
		public:
			DenseLayer(const DenseLayer& other);
			DenseLayer(size_t inputSize, size_t outputSize, fn::ActivationFunction&& activationFn, _DenseInitType initType = _DenseInitType::RANDOM, bool useBias = true);
			DenseLayer(Tensor&& weights, Tensor&& bias, fn::ActivationFunction&& activationFn);
			virtual const Tensor& propogate(const Tensor& input) override;
			virtual Tensor propogate(const Tensor& input) const override;
			virtual void backpropogate(const Tensor& pdToOut) override;
			virtual void gradientDescent(float learningRate, float momentumRate) override;
			virtual std::vector<size_t> getInputShape() const override { return { weights.getShape(0) }; }
			virtual std::vector<size_t> getOutputShape() const override { return { weights.getShape(1) }; }
			const Tensor& getWeights() const { return weights; }
			const Tensor& getBias() const { return bias; }
			const fn::ActivationFunction& getActivationFunction() const { return activationFn; }
			virtual void print() const override;
			virtual std::shared_ptr<Layer> clone() const override;

		private:
			Tensor weights;
			Tensor bias;
			fn::ActivationFunction activationFn;

			Tensor const* propogateInput = nullptr;
			Tensor pdToWeights;
			Tensor pdToBias;
			Tensor momentumWeights;
			Tensor momentumBias;
		};

		class ConvLayer : public Layer
		{
		public:
			ConvLayer(std::vector<size_t> kernel, std::vector<size_t> stride, fn::ActivationFunction&& activationFn);
			virtual const Tensor& propogate(const Tensor& input) override;
			virtual Tensor propogate(const Tensor& input) const override;
			virtual void backpropogate(const Tensor& pdToOut) override;
			virtual void gradientDescent(float learningRate, float momentumRate) override;
			virtual std::shared_ptr<Layer> clone() const override;

		private:
			fn::ActivationFunction activationFn;
		};

		class FlattenLayer : public Layer
		{
		public:
			FlattenLayer();
			virtual const Tensor& propogate(const Tensor& input) override;
			virtual Tensor propogate(const Tensor& input) const override;
			virtual void backpropogate(const Tensor& pdToOut) override;
			virtual std::shared_ptr<Layer> clone() const override;
		};

		class MaxPoolLayer : public Layer
		{
		public:
			MaxPoolLayer();
			virtual const Tensor& propogate(const Tensor& input) override;
			virtual Tensor propogate(const Tensor& input) const override;
			virtual void backpropogate(const Tensor& pdToOut) override;
			virtual std::shared_ptr<Layer> clone() const override;
		};
	}
};
