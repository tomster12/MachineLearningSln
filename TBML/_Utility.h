#pragma once

#include "stdafx.h"
#include "_Tensor.h"
#include <cassert>

namespace tbml
{
	namespace fn
	{
		float _classificationAccuracy(const tbml::_Tensor& predicted, const tbml::_Tensor& expected);

		class _ActivationFunction
		{
		public:
			_ActivationFunction() : activateFn(nullptr), chainDerivativeFn(nullptr) {}
			virtual void activate(_Tensor& x) const { activateFn(x); }
			virtual _Tensor chainDerivative(const _Tensor& z, const _Tensor& pdToOut) const { return chainDerivativeFn(z, pdToOut); }

		private:
			std::function<void(_Tensor&)> activateFn;
			std::function<_Tensor(const _Tensor&, const _Tensor&)> chainDerivativeFn;

		protected:
			_ActivationFunction(
				std::function<void(_Tensor& x)> activateFn,
				std::function<_Tensor(const _Tensor& z, const _Tensor& pdToOut)> chainDerivativeFn)
				: activateFn(activateFn), chainDerivativeFn(chainDerivativeFn)
			{}
		};

		class _LossFunction
		{
		public:
			_LossFunction() : lossFn(nullptr), derivativeFn(nullptr) {}
			virtual float activate(const _Tensor& predicted, const _Tensor& expected) const { return lossFn(predicted, expected); }
			virtual _Tensor derive(const _Tensor& predicted, const _Tensor& expected) const { return derivativeFn(predicted, expected); }

		private:
			std::function<float(const _Tensor& predicted, const _Tensor& expected)> lossFn;
			std::function<_Tensor(const _Tensor& predicted, const _Tensor& expected)> derivativeFn;

		protected:
			_LossFunction(
				std::function<float(const _Tensor& predicted, const _Tensor& expected)> lossFn,
				std::function<_Tensor(const _Tensor& predicted, const _Tensor& expected)> derivativeFn)
				: lossFn(lossFn), derivativeFn(derivativeFn)
			{}
		};

		class _ReLU : public _ActivationFunction
		{
		public:
			_ReLU() : _ActivationFunction(
				[](_Tensor& x)
			{
				x.map([](float x) { return std::max(0.0f, x); });
			},
				[](const _Tensor& z, const _Tensor& pdToOut)
			{
				_Tensor pdOutToIn = z.mapped([](float v) { return v > 0 ? 1.0f : 0.0f; });
				return pdOutToIn * pdToOut;
			})
			{}
		};

		class _Sigmoid : public _ActivationFunction
		{
		public:
			_Sigmoid() : _ActivationFunction(
				[this](_Tensor& x)
			{
				x.map([this](float x) { return sigmoid(x); });
			},
				[this](const _Tensor& z, const _Tensor& pdToOut)
			{
				_Tensor pdOutToIn = z.mapped([this](float v)
				{
					float sv = sigmoid(v);
					return sv * (1.0f - sv);
				});

				return pdOutToIn * pdToOut;
			})
			{}

		private:
			float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
		};

		class _TanH : public _ActivationFunction
		{
		public:
			_TanH() : _ActivationFunction(
				[](_Tensor& x)
			{
				x.map([](float x) { return tanhf(x); });
			},
				[](const _Tensor& z, const _Tensor& pdToOut)
			{
				_Tensor pdOutToIn = z.mapped([](float v)
				{
					float th = tanhf(v);
					return 1 - th * th;
				});

				return pdOutToIn * pdToOut;
			})
			{}
		};

		class _SoftMax : public _ActivationFunction
		{
		public:
			_SoftMax() : _ActivationFunction(
				[this](_Tensor& x)
			{
				auto shape = x.getShape();
				assert(shape.size() == 2);

				// Independent per row
				for (size_t row = 0; row < shape[0]; row++)
				{
					// Calculate max of row for stability
					float max = x(row, 0);
					for (size_t i = 1; i < shape[1]; i++) max = std::max(max, x(row, i));

					// SoftMax of each element in row = e^(X(i) - max) / Σ e^(X(i) - max)
					float sum = 0.0;
					for (size_t i = 0; i < shape[1]; i++)
					{
						x(row, i) = std::exp(x(row, i) - max);
						sum += x(row, i);
					}
					for (size_t i = 0; i < shape[1]; i++) x(row, i) /= sum;
				}
			},
				[this](const _Tensor& z, const _Tensor& pdToOut)
			{
				auto shape = z.getShape();
				assert(shape.size() == 2);
				_Tensor result(shape, 0);

				// Independent per row
				for (size_t row = 0; row < shape[0]; row++)
				{
					// For each neuron i
					for (size_t i = 0; i < shape[1]; i++)
					{
						float Zi = z(row, i);

						// Sum partial derivatives Zj / Xi
						for (size_t j = 0; j < shape[1]; j++)
						{
							float Zj = z(row, j);
							int kronekerDelta = (i == j) ? 1 : 0;
							float dSij = (Zj * (kronekerDelta - Zi));
							result(row, i) += dSij * pdToOut(row, j);
						}
					}
				}

				return result;
			})
			{}
		};

		class _SquareError : public _LossFunction
		{
		public:
			_SquareError() : _LossFunction(
				[](const _Tensor& predicted, const _Tensor& expected)
			{
				const auto& predicteddata = predicted.getData();
				const auto& expecteddata = expected.getData();
				assert(predicteddata.size() == expecteddata.size());

				float error = 0.0;
				for (size_t i = 0; i < predicteddata.size(); i++)
				{
					float diff = predicteddata[i] - expecteddata[i];
					error += diff * diff;
				}
				return error;
			},
				[](const _Tensor& predicted, const _Tensor& expected)
			{
				assert(predicted.getShape() == expected.getShape());

				// derivative of square error = YH - Y
				return predicted - expected;
			})
			{}
		};

		class _CrossEntropy : public _LossFunction
		{
		public:
			_CrossEntropy() : _LossFunction(
				[](const _Tensor& predicted, const _Tensor& expected)
			{
				const auto& predictedData = predicted.getData();
				const auto& expectedData = expected.getData();
				assert(predictedData.size() == expectedData.size());

				// Cross entropy = Σ -Yi * log(YHi + e) with epsilon = 1e-15f for stability
				float error = 0.0;
				for (size_t i = 0; i < predictedData.size(); i++)
				{
					error += -expectedData[i] * std::log(predictedData[i] + float(1e-15f));
				}
				return error / predicted.getShape()[0];
			},
				[](const _Tensor& predicted, const _Tensor& expected)
			{
				assert(predicted.getShape() == expected.getShape());

				// derivative of cross entropy = Yi / (YHi + e) with epsilon = 1e-15f for stability
				return expected.ewised(predicted, [](float expected, float predicted) { return -expected / (predicted + 1e-15f); });
			})
			{}
		};
	};
}
