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
			size_t epochs = 20;
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
			_NeuralNetwork(fn::_LossFunction&& lossFn);
			_NeuralNetwork(fn::_LossFunction&& lossFn, std::vector<_Layer*> layers);
			~_NeuralNetwork();
			void addLayer(_Layer* layer);
			const _Tensor& propogate(const _Tensor& input);
			void train(const std::vector<_Tensor>& inputs, const std::vector<_Tensor>& expectedOutputs, const _TrainingConfig& config);

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
			const _Tensor& getPropogateOutput() const { return propogateOutput; }
			const _Tensor& getBackpropogateOutput() const { return backpropogateOutput; }

		protected:
			_Tensor propogateOutput;
			_Tensor backpropogateOutput;
		};

		enum _DenseInitType { ZERO, RANDOM };

		class _DenseLayer : public _Layer
		{
		public:
			_DenseLayer(size_t inputSize, size_t outputSize, fn::_ActivationFunction&& actFn, _DenseInitType initType = _DenseInitType::RANDOM, bool useBias = true);
			virtual void propogate(const _Tensor& input) override;
			virtual void backpropogate(const _Tensor& pdToOut) override;

		private:
			_Tensor weights;
			_Tensor bias;
			fn::_ActivationFunction actFn;
		};

		class _ConvLayer : public _Layer
		{
		public:
			_ConvLayer(std::vector<size_t> kernel, std::vector<size_t> stride, fn::_ActivationFunction&& actFn);
			virtual void propogate(const _Tensor& input) override;
			virtual void backpropogate(const _Tensor& pdToOut) override;

		private:
			fn::_ActivationFunction actFn;
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
