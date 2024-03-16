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

			Layer() = default;
			virtual ~Layer() = default;
			Layer(const Layer&) = delete;
			Layer& operator=(const Layer&) = delete;
			Layer(Layer&&) = delete;
			Layer& operator=(Layer&&) = delete;

			virtual const Tensor& propogate(const Tensor& input) = 0;
			virtual Tensor propogate(const Tensor& input) const = 0;
			virtual void propogateMut(Tensor& input) const = 0;
			virtual void backpropogate(const Tensor& pdToOut) = 0;
			virtual void gradientDescent(float learningRate, float momentumRate) {};
			virtual std::vector<size_t> getInputShape() const { return {}; }; // TODO: Make pure virtual
			virtual std::vector<size_t> getOutputShape() const { return {}; }; // TODO: Make pure virtual
			const Tensor& getOutput() const { return output; };
			const Tensor& getPdToIn() const { return pdToIn; };
			virtual int getParameterCount() const { return 0; };
			virtual void print() const {};
			virtual std::shared_ptr<Layer> clone() const = 0;

			virtual void serialize(std::ostream& os) const = 0;
			static std::shared_ptr<Layer> deserialize(std::istream& is);

		protected:
			Tensor output;
			Tensor pdToIn;
		};

		using LayerPtr = std::shared_ptr<Layer>;

		class NeuralNetwork
		{
		public:
			static const int MAX_EPOCHS = 1'000;

			NeuralNetwork() {}
			NeuralNetwork(fn::LossFunctionPtr&& lossFn) : lossFn(std::move(lossFn)) {}
			NeuralNetwork(fn::LossFunctionPtr&& lossFn, std::vector<LayerPtr>&& layers) : lossFn(std::move(lossFn)), layers(std::move(layers)) {}

			void addLayer(LayerPtr&& layer);
			const Tensor& propogate(const Tensor& input);
			Tensor propogate(const Tensor& input) const;
			void train(const Tensor& input, const Tensor& expected, const TrainingConfig& config);
			std::vector<size_t> getInputShape() const { return layers[0]->getInputShape(); }
			std::vector<size_t> getOutputShape() const { return layers[layers.size() - 1]->getOutputShape(); }
			const std::vector<LayerPtr>& getLayers() const { return layers; }
			fn::LossFunctionPtr getLossFunction() const { return lossFn; }
			int getParameterCount() const;
			void print() const;

			void saveToFile(const std::string& filename) const;
			static NeuralNetwork loadFromFile(const std::string& filename);

		private:
			fn::LossFunctionPtr lossFn;
			std::vector<LayerPtr> layers;
		};

		enum class _DenseInitType { ZERO, RANDOM };

		class DenseLayer : public Layer
		{
		public:

			DenseLayer(const DenseLayer& other);
			DenseLayer(size_t inputSize, size_t outputSize, fn::ActivationFunctionPtr&& activationFn, _DenseInitType initType = _DenseInitType::RANDOM, bool useBias = true);
			DenseLayer(Tensor&& weights, Tensor&& bias, fn::ActivationFunctionPtr&& activationFn);

			const Tensor& propogate(const Tensor& input) override;
			Tensor propogate(const Tensor& input) const override;
			void propogateMut(Tensor& input) const override;
			void backpropogate(const Tensor& pdToOut) override;
			void gradientDescent(float learningRate, float momentumRate) override;
			std::vector<size_t> getInputShape() const override { return { weights.getShape(0) }; }
			std::vector<size_t> getOutputShape() const override { return { weights.getShape(1) }; }
			int getParameterCount() const override { return weights.getSize() + bias.getSize(); }
			const Tensor& getWeights() const { return weights; }
			const Tensor& getBias() const { return bias; }
			const fn::ActivationFunctionPtr& getActivationFunction() const { return activationFn; }
			virtual void print() const override;
			virtual LayerPtr clone() const override;
			virtual void serialize(std::ostream& os) const override;

		private:
			Tensor weights;
			Tensor bias;
			fn::ActivationFunctionPtr activationFn;

			Tensor const* propogateInput = nullptr;
			Tensor pdToWeights;
			Tensor pdToBias;
			Tensor momentumWeights;
			Tensor momentumBias;
		};

		/*
		class ConvLayer : public Layer
		{
		public:
			ConvLayer(std::vector<size_t> kernel, std::vector<size_t> stride, fn::ActivationFunction&& activationFn);
			virtual const Tensor& propogate(const Tensor& input) override;
			virtual Tensor propogate(const Tensor& input) const override;
			virtual void propogate(Tensor& input) const override;
			virtual void backpropogate(const Tensor& pdToOut) override;
			virtual void gradientDescent(float learningRate, float momentumRate) override;
			virtual LayerPtr clone() const override;

		private:
			fn::ActivationFunction activationFn;
		};

		class FlattenLayer : public Layer
		{
		public:
			FlattenLayer();
			virtual const Tensor& propogate(const Tensor& input) override;
			virtual Tensor propogate(const Tensor& input) const override;
			virtual void propogate(Tensor& input) const override;
			virtual void backpropogate(const Tensor& pdToOut) override;
			virtual LayerPtr clone() const override;
		};

		class MaxPoolLayer : public Layer
		{
		public:
			MaxPoolLayer();
			virtual const Tensor& propogate(const Tensor& input) override;
			virtual Tensor propogate(const Tensor& input) const override;
			virtual void propogate(Tensor& input) const override;
			virtual void backpropogate(const Tensor& pdToOut) override;
			virtual LayerPtr clone() const override;
		};
		*/
	}
};
