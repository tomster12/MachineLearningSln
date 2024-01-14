#pragma once

#include "stdafx.h"
#include "Matrix.h"

namespace tbml
{
	namespace fn
	{
		float getRandomFloat();
		float calculateAccuracy(tbml::Matrix const& predicted, tbml::Matrix const& expected);

		class ActivationFunction
		{
		public:
			ActivationFunction() : activateFn(nullptr), chainDerivativeFn(nullptr) {}
			virtual void operator()(Matrix& x) const { activateFn(x); }
			virtual Matrix derivative(Matrix const& x, Matrix const& pdToOut) const { return chainDerivativeFn(x, pdToOut); }

		private:
			std::function<void(Matrix&)> activateFn;
			std::function<Matrix(Matrix const&, Matrix const&)> chainDerivativeFn;

		protected:
			ActivationFunction(
				std::function<void(Matrix&)> activate,
				std::function<Matrix(Matrix const&, Matrix const&)> chainDerivativeFn)
				: activateFn(activate), chainDerivativeFn(chainDerivativeFn)
			{}
		};

		class ErrorFunction
		{
		public:
			ErrorFunction() : errorFn(nullptr), derivativeFn(nullptr) {}
			virtual float operator()(Matrix const& predicted, Matrix const& expected) const { return errorFn(predicted, expected); }
			virtual Matrix derivative(Matrix const& predicted, Matrix const& expected) const { return derivativeFn(predicted, expected); }

		private:
			std::function<float(Matrix const& predicted, Matrix const& expected)> errorFn;
			std::function<Matrix(Matrix const& predicted, Matrix const& expected)> derivativeFn;

		protected:
			ErrorFunction(
				std::function<float(Matrix const& predicted, Matrix const& expected)> errorFn,
				std::function<Matrix(Matrix const& predicted, Matrix const& expected)> derivativeFn)
				: errorFn(errorFn), derivativeFn(derivativeFn)
			{}
		};

		class ReLU : public ActivationFunction
		{
		public:
			ReLU() : ActivationFunction(
				[](Matrix& x) { x.map([](float x) { return std::max(0.0f, x); }); },
				[](Matrix const& x, Matrix const& pdToOut) { return x.mapped([](float v) { return v > 0 ? 1.0f : 0.0f; }) * pdToOut; })
			{}
		};

		class Sigmoid : public ActivationFunction
		{
		public:
			Sigmoid() : ActivationFunction(
				[this](Matrix& x) { x.map([this](float x) { return sigmoid(x); }); },
				[this](Matrix const& x, Matrix const& pdToOut) { return x.mapped([this](float x) { return sigmoid(x) * (1.0f - sigmoid(x)); }) * pdToOut; })
			{}

		private:
			float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
		};

		class TanH : public ActivationFunction
		{
		public:
			TanH() : ActivationFunction(
				[](Matrix& x) { x.map([](float x) { return tanhf(x); }); },
				[](Matrix const& x, Matrix const& pdToOut) { return x.mapped([](float x) { float th = tanhf(x); return 1 - th * th; }) * pdToOut; })
			{}
		};

		class SoftMax : public ActivationFunction
		{
		public:
			SoftMax() : ActivationFunction(
				[this](Matrix& x) { softmax(x); },
				[this](Matrix const& s, Matrix const& pdToOut)
			{
				size_t rows = s.getRowCount();
				size_t cols = s.getColCount();
				Matrix result(rows, cols);

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
			void softmax(Matrix& x)
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

		class SquareError : public ErrorFunction
		{
		public:
			SquareError() : ErrorFunction(
				[](Matrix const& predicted, Matrix const& expected)
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
				[](Matrix const& predicted, Matrix const& expected)
			{
				return predicted - expected;
			})
			{}
		};

		class CrossEntropy : public ErrorFunction
		{
		public:
			CrossEntropy() : ErrorFunction(
				[](Matrix const& predicted, Matrix const& expected)
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
				[](Matrix const& predicted, Matrix const& expected)
			{
				size_t rows = expected.getRowCount();
				size_t cols = expected.getColCount();

				Matrix grad(rows, cols);
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
