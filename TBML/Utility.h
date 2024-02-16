#pragma once

#include "stdafx.h"
#include "Matrix.h"

namespace tbml
{
	namespace fn
	{
		float getRandomFloat();
		float calculateAccuracy(tbml::Tensor const& predicted, tbml::Tensor const& expected);

		class ActivationFunction
		{
		public:
			ActivationFunction() : activateFn(nullptr), chainDerivativeFn(nullptr) {}
			virtual void operator()(Tensor& x) const { activateFn(x); }
			virtual Tensor derivative(Tensor const& x, Tensor const& pdToOut) const { return chainDerivativeFn(x, pdToOut); }

		private:
			std::function<void(Tensor&)> activateFn;
			std::function<Tensor(Tensor const&, Tensor const&)> chainDerivativeFn;

		protected:
			ActivationFunction(
				std::function<void(Tensor&)> activate,
				std::function<Tensor(Tensor const&, Tensor const&)> chainDerivativeFn)
				: activateFn(activate), chainDerivativeFn(chainDerivativeFn)
			{}
		};

		class LossFunction
		{
		public:
			LossFunction() : errorFn(nullptr), derivativeFn(nullptr) {}
			virtual float operator()(Tensor const& predicted, Tensor const& expected) const { return errorFn(predicted, expected); }
			virtual Tensor derivative(Tensor const& predicted, Tensor const& expected) const { return derivativeFn(predicted, expected); }

		private:
			std::function<float(Tensor const& predicted, Tensor const& expected)> errorFn;
			std::function<Tensor(Tensor const& predicted, Tensor const& expected)> derivativeFn;

		protected:
			LossFunction(
				std::function<float(Tensor const& predicted, Tensor const& expected)> errorFn,
				std::function<Tensor(Tensor const& predicted, Tensor const& expected)> derivativeFn)
				: errorFn(errorFn), derivativeFn(derivativeFn)
			{}
		};

		class ReLU : public ActivationFunction
		{
		public:
			ReLU() : ActivationFunction(
				[](Tensor& x) { x.map([](float x) { return std::max(0.0f, x); }); },
				[](Tensor const& x, Tensor const& pdToOut) { return x.mapped([](float v) { return v > 0 ? 1.0f : 0.0f; }) * pdToOut; })
			{}
		};

		class Sigmoid : public ActivationFunction
		{
		public:
			Sigmoid() : ActivationFunction(
				[this](Tensor& x) { x.map([this](float x) { return sigmoid(x); }); },
				[this](Tensor const& x, Tensor const& pdToOut) { return x.mapped([this](float x) { return sigmoid(x) * (1.0f - sigmoid(x)); }) * pdToOut; })
			{}

		private:
			float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
		};

		class TanH : public ActivationFunction
		{
		public:
			TanH() : ActivationFunction(
				[](Tensor& x) { x.map([](float x) { return tanhf(x); }); },
				[](Tensor const& x, Tensor const& pdToOut) { return x.mapped([](float x) { float th = tanhf(x); return 1 - th * th; }) * pdToOut; })
			{}
		};

		class SoftMax : public ActivationFunction
		{
		public:
			SoftMax() : ActivationFunction(
				[this](Tensor& x) { softmax(x); },
				[this](Tensor const& s, Tensor const& pdToOut)
			{
				size_t rows = s.getRowCount();
				size_t cols = s.getColCount();
				Tensor result(rows, cols);

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
			})
			{}

		private:
			void softmax(Tensor& x)
			{
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
				}
			};
		};

		class SquareError : public LossFunction
		{
		public:
			SquareError() : LossFunction(
				[](Tensor const& predicted, Tensor const& expected)
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
				[](Tensor const& predicted, Tensor const& expected)
			{
				return predicted - expected;
			})
			{}
		};

		class CrossEntropy : public LossFunction
		{
		public:
			CrossEntropy() : LossFunction(
				[](Tensor const& predicted, Tensor const& expected)
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
				[](Tensor const& predicted, Tensor const& expected)
			{
				size_t rows = expected.getRowCount();
				size_t cols = expected.getColCount();

				Tensor grad(rows, cols);
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
