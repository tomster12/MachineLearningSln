#pragma once

#include "Utility.h"
#include "Tensor.h"

namespace tbml
{
	namespace nn
	{
		namespace Layer
		{
			class Base
			{
			public:
				Base() = default;
				virtual ~Base() = default;
				Base(const Base&) = delete;
				Base& operator=(const Base&) = delete;
				Base(Base&&) = delete;
				Base& operator=(Base&&) = delete;

				virtual void propogateMut(Tensor& input) const = 0;
				virtual const Tensor& propogateRef(const Tensor& input) = 0;
				virtual void backpropogate(const Tensor& gradOutput) = 0;
				virtual void gradientDescent(float learningRate, float momentumRate) {};
				virtual void print() const {};
				virtual std::shared_ptr<Base> clone() const = 0;

				void setRetainValues(bool retainValues) { this->retainValues = retainValues; }
				virtual std::vector<size_t> getInputShape() const = 0;
				virtual std::vector<size_t> getOutputShape() const = 0;
				const Tensor& getOutput() const { return output; };
				const Tensor& getGradInput() const { return gradInput; };
				virtual int getParameterCount() const { return 0; };

				virtual void serialize(std::ostream& os) const = 0;

			protected:
				Tensor output;
				Tensor gradInput;
				bool retainValues = false;
			};

			using BasePtr = std::shared_ptr<Base>;

			static BasePtr deserialize(std::istream& is);

			enum class DenseInitType { ZERO, RANDOM };

			class Dense : public Base
			{
			public:

				Dense(const Dense& other);
				Dense(size_t inputSize, size_t outputSize, fn::ActivationFunctionPtr&& activationFn, DenseInitType initType = DenseInitType::RANDOM, bool useBias = true);
				Dense(Tensor&& weights, Tensor&& bias, fn::ActivationFunctionPtr&& activationFn);

				virtual void propogateMut(Tensor& input) const override;
				virtual const Tensor& propogateRef(const Tensor& input) override;
				void backpropogate(const Tensor& gradOutput) override;
				void gradientDescent(float learningRate, float momentumRate) override;
				virtual void print() const override;
				virtual BasePtr clone() const override;

				std::vector<size_t> getInputShape() const override { return { weights.getShape(0) }; }
				std::vector<size_t> getOutputShape() const override { return { weights.getShape(1) }; }
				int getParameterCount() const override { return weights.getSize() + bias.getSize(); }
				const Tensor& getWeights() const { return weights; }
				const Tensor& getBias() const { return bias; }
				const fn::ActivationFunctionPtr& getActivationFunction() const { return activationFn; }

				virtual void serialize(std::ostream& os) const override;

			private:
				Tensor weights;
				Tensor bias;
				fn::ActivationFunctionPtr activationFn;

				Tensor const* propogateInput = nullptr;
				Tensor gradWeights;
				Tensor gradBias;
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
				virtual void backpropogate(const Tensor& gradOutput) override;
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
				virtual void backpropogate(const Tensor& gradOutput) override;
				virtual LayerPtr clone() const override;
			};

			class MaxPoolLayer : public Layer
			{
			public:
				MaxPoolLayer();
				virtual const Tensor& propogate(const Tensor& input) override;
				virtual Tensor propogate(const Tensor& input) const override;
				virtual void propogate(Tensor& input) const override;
				virtual void backpropogate(const Tensor& gradOutput) override;
				virtual LayerPtr clone() const override;
			};
			*/
		}

		struct TrainingConfig
		{
			int epochs = 20;
			int batchSize = -1;
			float learningRate = 0.1f;
			float momentumRate = 0.1f;
			float errorThreshold = 0.0f;
			size_t logLevel = 0;
		};

		class NeuralNetwork
		{
		public:
			static const int MAX_EPOCHS = 1'000;

			NeuralNetwork() {}
			NeuralNetwork(fn::LossFunctionPtr&& lossFn) : lossFn(std::move(lossFn)) {}
			NeuralNetwork(fn::LossFunctionPtr&& lossFn, std::vector<Layer::BasePtr>&& layers) : lossFn(std::move(lossFn)), layers(std::move(layers)) {}

			void addLayer(Layer::BasePtr&& layer);
			virtual Tensor propogate(const Tensor& input) const;
			virtual void propogateMut(Tensor& input) const;
			virtual const Tensor& propogateRef(const Tensor& input);
			void train(const Tensor& input, const Tensor& expected, const TrainingConfig& config);
			void print() const;

			std::vector<size_t> getInputShape() const { return layers[0]->getInputShape(); }
			std::vector<size_t> getOutputShape() const { return layers[layers.size() - 1]->getOutputShape(); }
			const std::vector<Layer::BasePtr>& getLayers() const { return layers; }
			fn::LossFunctionPtr getLossFunction() const { return lossFn; }
			int getParameterCount() const;

			void saveToFile(const std::string& filename) const;
			static NeuralNetwork loadFromFile(const std::string& filename);

		private:
			fn::LossFunctionPtr lossFn;
			std::vector<Layer::BasePtr> layers;
		};
	}
};
