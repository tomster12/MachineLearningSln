#pragma once

#include "Utility.h"
#include "_Tensor.h"

namespace tbml
{
	namespace nn
	{
		class _Layer
		{
		public:
			virtual const _Tensor& propogate(const _Tensor& input) = 0;
			virtual const _Tensor& backpropagate(const _Tensor& dToOut) = 0;
			const _Tensor& getPropogateOutput() { return propogateOutput; }
			const _Tensor& getBackpropogateOutput() { return backpropagateOutput; }

		protected:
			_Tensor propogateOutput;
			_Tensor backpropagateOutput;
		};

		struct _TrainingConfig
		{
			size_t epochs = 20;
			int batchSize = -1;
			float learningRate = 0.1f;
			float momentumRate = 0.1f;
			float errorThreshold = 0.0f;
			size_t logLevel = 0;
		};

		class _NeuralNetwork
		{
		public:
			_NeuralNetwork(fn::LossFunction&& lossFn);
			_NeuralNetwork(fn::LossFunction&& lossFn, std::vector<_Layer*>&& layers);
			void addLayer(_Layer* layer);
			const _Tensor& propogate(const _Tensor& input);
			const _Tensor& train(const std::vector<_Tensor>& inputs, const std::vector<_Tensor>& expectedOutputs, const _TrainingConfig& config);

		private:
			fn::LossFunction lossFn;
			std::vector<_Layer*> layers;

			void backpropogate(const _Tensor& loss) const;
		};

		enum _DenseInitType { ZERO, RANDOM };

		class _DenseLayer : _Layer
		{
		public:
			_DenseLayer(size_t inputSize, size_t outputSize, fn::ActivationFunction&& actFn, _DenseInitType initType = _DenseInitType::RANDOM, bool useBias = true);
			virtual const _Tensor& propogate(const _Tensor& input) override;
			virtual const _Tensor& backpropagate(const _Tensor& dToOut) override;

		private:
			_Tensor weights;
			_Tensor bias;
			fn::ActivationFunction actFn;
		};

		class _ConvLayer : _Layer
		{
		public:
			_ConvLayer(std::vector<size_t> kernel, std::vector<size_t> stride, fn::ActivationFunction&& actFn);
			virtual const _Tensor& propogate(const _Tensor& input) override;
			virtual const _Tensor& backpropagate(const _Tensor& dToOut) override;

		private:
			fn::ActivationFunction actFn;
		};

		class _FlattenLayer : _Layer
		{
		public:
			_FlattenLayer();
			virtual const _Tensor& propogate(const _Tensor& input) override;
			virtual const _Tensor& backpropagate(const _Tensor& dToOut) override;
		};

		class _MaxPoolLayer : _Layer
		{
		public:
			_MaxPoolLayer();
			virtual const _Tensor& propogate(const _Tensor& input) override;
			virtual const _Tensor& backpropagate(const _Tensor& dToOut) override;
		};
	}
};
