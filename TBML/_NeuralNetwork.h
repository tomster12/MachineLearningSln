#pragma once

#include "Utility.h"
#include "_Utility.h"
#include "_Tensor.h"

namespace tbml
{
	namespace nn
	{
		struct _TrainingConfig
		{
			int epochs = 20;
			int batchSize = -1;
			float learningRate = 0.1f;
			float momentumRate = 0.1f;
			float errorThreshold = 0.0f;
			size_t logLevel = 0;
		};

		class _Layer;
		class _NeuralNetwork
		{
		public:
			static const int MAX_EPOCHS = 1'000;

			_NeuralNetwork(fn::_LossFunction&& lossFn);
			_NeuralNetwork(fn::_LossFunction&& lossFn, std::vector<_Layer*> layers);
			~_NeuralNetwork();
			void addLayer(_Layer* layer);
			const _Tensor& propogate(const _Tensor& input);
			void train(const _Tensor& input, const _Tensor& expected, const _TrainingConfig& config);
			void print() const;

		private:
			fn::_LossFunction lossFn;
			std::vector<_Layer*> layers;

			void backpropogate(const _Tensor& predicted, const _Tensor& expected);
		};

		class _Layer
		{
		public:
			virtual void propogate(const _Tensor& input) = 0;
			virtual void backpropogate(const _Tensor& pdToOut) = 0;
			virtual void gradientDescent(float learningRate, float momentumRate) {};
			const _Tensor& getPredicted() const { return predicted; }
			const _Tensor& getPdToIn() const { return pdToIn; }
			virtual void print() const {};

		protected:
			_Tensor predicted;
			_Tensor pdToIn;
		};

		enum class _DenseInitType { ZERO, RANDOM };

		class _DenseLayer : public _Layer
		{
		public:
			_DenseLayer(size_t inputSize, size_t outputSize, fn::_ActivationFunction&& activationFn, _DenseInitType initType = _DenseInitType::RANDOM, bool useBias = true);
			virtual void propogate(const _Tensor& input) override;
			virtual void backpropogate(const _Tensor& pdToOut) override;
			virtual void gradientDescent(float learningRate, float momentumRate) override;
			virtual void print() const override;

		private:
			_Tensor weights;
			_Tensor bias;
			fn::_ActivationFunction activationFn;

			_Tensor const* propogateInput;
			_Tensor pdToWeights;
			_Tensor pdToBias;
			_Tensor momentumWeights;
			_Tensor momentumBias;
		};

		class _ConvLayer : public _Layer
		{
		public:
			_ConvLayer(std::vector<size_t> kernel, std::vector<size_t> stride, fn::_ActivationFunction&& activationFn);
			virtual void propogate(const _Tensor& input) override;
			virtual void backpropogate(const _Tensor& pdToOut) override;
			virtual void gradientDescent(float learningRate, float momentumRate) override;

		private:
			fn::_ActivationFunction activationFn;
		};

		class _FlattenLayer : public _Layer
		{
		public:
			_FlattenLayer();
			virtual void propogate(const _Tensor& input) override;
			virtual void backpropogate(const _Tensor& pdToOut) override;
		};

		class _MaxPoolLayer : public _Layer
		{
		public:
			_MaxPoolLayer();
			virtual void propogate(const _Tensor& input) override;
			virtual void backpropogate(const _Tensor& pdToOut) override;
		};
	}
};
