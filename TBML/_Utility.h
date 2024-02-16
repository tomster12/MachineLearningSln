#pragma once

#include "stdafx.h"
#include "_Tensor.h"

namespace tbml
{
	namespace fn
	{
		class _ActivationFunction
		{
		public:
			_ActivationFunction() : activateFn(nullptr), chainDerivativeFn(nullptr) {}
			virtual void operator()(_Tensor& x) const { activateFn(x); }
			virtual _Tensor derivative(_Tensor const& x, _Tensor const& pdToOut) const { return chainDerivativeFn(x, pdToOut); }

		private:
			std::function<void(_Tensor&)> activateFn;
			std::function<_Tensor(_Tensor const&, _Tensor const&)> chainDerivativeFn;

		protected:
			_ActivationFunction(
				std::function<void(_Tensor&)> activate,
				std::function<_Tensor(_Tensor const&, _Tensor const&)> chainDerivativeFn)
				: activateFn(activate), chainDerivativeFn(chainDerivativeFn)
			{}
		};

		class _LossFunction
		{
		public:
			_LossFunction() : errorFn(nullptr), derivativeFn(nullptr) {}
			virtual float operator()(_Tensor const& predicted, _Tensor const& expected) const { return errorFn(predicted, expected); }
			virtual _Tensor derivative(_Tensor const& predicted, _Tensor const& expected) const { return derivativeFn(predicted, expected); }

		private:
			std::function<float(_Tensor const& predicted, _Tensor const& expected)> errorFn;
			std::function<_Tensor(_Tensor const& predicted, _Tensor const& expected)> derivativeFn;

		protected:
			_LossFunction(
				std::function<float(_Tensor const& predicted, _Tensor const& expected)> errorFn,
				std::function<_Tensor(_Tensor const& predicted, _Tensor const& expected)> derivativeFn)
				: errorFn(errorFn), derivativeFn(derivativeFn)
			{}
		};

		class ReLU : public _ActivationFunction
		{
		public:
			ReLU() : _ActivationFunction(
				[](_Tensor& x) { x.map([](float x) { return std::max(0.0f, x); }); },
				[](_Tensor const& x, _Tensor const& pdToOut) { return x.mapped([](float v) { return v > 0 ? 1.0f : 0.0f; }) * pdToOut; })
			{}
		};

		class Sigmoid : public _ActivationFunction
		{
		public:
			Sigmoid() : _ActivationFunction(
				[this](_Tensor& x) { x.map([this](float x) { return sigmoid(x); }); },
				[this](_Tensor const& x, _Tensor const& pdToOut) { return x.mapped([this](float x) { return sigmoid(x) * (1.0f - sigmoid(x)); }) * pdToOut; })
			{}

		private:
			float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
		};

		class TanH : public _ActivationFunction
		{
		public:
			TanH() : _ActivationFunction(
				[](_Tensor& x) { x.map([](float x) { return tanhf(x); }); },
				[](_Tensor const& x, _Tensor const& pdToOut) { return x.mapped([](float x) { float th = tanhf(x); return 1 - th * th; }) * pdToOut; })
			{}
		};

		class SoftMax : public _ActivationFunction
		{
		public:
			SoftMax() : _ActivationFunction(
				[this](_Tensor& x)
			{
				/*
				size_t rows = x.getRowCount();
				size_t cols = x.getColCount();

				// Consider each row independantly
				for (size_t row = 0; row < rows; row++)
				{
					// Calculate max for stability
					float max = x(row, 0);
					for (size_t col = 1; col < cols; col++) max = std::max(max, x(row, col));

					// SoftMax = Sⱼ = eˣᶦ⁺ᴰ / Σ(Cᵃʲ⁺ᴰ), With D = -max for stability
					float sum = 0.0;
					for (size_t i = 0; i < cols; i++)
					{
						x(row, i) = std::exp(x(row, i) - max);
						sum += x(row, i);
					}
					for (size_t j = 0; j < cols; j++) x(row, j) /= sum;
				}*/
			},
			[this](_Tensor const& s, _Tensor const& pdToOut)
			{
				size_t rows = s.getRowCount();
				size_t cols = s.getColCount();
				_Tensor result(rows, cols);

				// Independent per row
				for (size_t row = 0; row < rows; row++)
				{
					// For each neuron input X(j)
					for (size_t j = 0; j < cols; j++)
					{
						float Sj = s(row, j);

						// Calculate derivative d S(j) / X(i)
						for (size_t i = 0; i < cols; i++)
						{
							float Si = s(row, i);
							int kronekerDelta = (j == i) ? 1 : 0;
							float dSij = (Si * (kronekerDelta - Sj));
							result(row, j) += dSij * pdToOut(row, i);
						}
					}
				}

				return result;
			});
		};

		class SquareError : public _LossFunction
		{
		public:
			SquareError() : _LossFunction(
				[](_Tensor const& predicted, _Tensor const& expected)
			{
				float error = 0;
				for (size_t row = 0; row < predicted.getRowCount(); row++)
				{
					// Sum of squared errors = Σ (Ei - Yi)^2
					for (size_t i = 0; i < predicted.getColCount(); i++)
					{
						float diff = expected(row, i) - predicted(row, i);
						error += diff * diff;
					}
				}
				return error;
			},
				[](_Tensor const& predicted, _Tensor const& expected)
			{
				return predicted - expected;
			})
			{}
		};

		class CrossEntropy : public _LossFunction
		{
		public:
			CrossEntropy() : _LossFunction(
				[](_Tensor const& predicted, _Tensor const& expected)
			{
				size_t rows = expected.getRowCount();
				size_t cols = expected.getColCount();

				float error = 0.0;
				for (size_t row = 0; row < rows; ++row)
				{
					// Categorical cross entropy = Σ Ei * log(Yi + e) with epsilon = 1e-15f for stability
					for (size_t i = 0; i < cols; i++)
					{
						error += -expected(row, i) * std::log(predicted(row, i) + float(1e-15f));
					}
				}

				return error / rows;
			},
				[](_Tensor const& predicted, _Tensor const& expected)
			{
				size_t rows = expected.getRowCount();
				size_t cols = expected.getColCount();

				_Tensor grad(rows, cols);
				for (size_t row = 0; row < rows; row++)
				{
					// Categorical cross entropy = Ei / (Yi + e) with epsilon = 1e-15f for stability
					for (size_t i = 0; i < cols; i++)
					{
						grad(row, i) = -expected(row, i) / (predicted(row, i) + 1e-15f);
					}
				}

				return grad;
			})
			{}
		};
	};
}
