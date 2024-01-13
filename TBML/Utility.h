#pragma once

#include "stdafx.h"
#include "Matrix.h"

namespace tbml
{
	namespace fns
	{
		float getRandomFloat();

		class ActivationFunction
		{
		public:
			ActivationFunction() : _activate(nullptr), _derive(nullptr) {}
			virtual void operator()(Matrix& x) const { _activate(x); }
			virtual Matrix derive(Matrix const& x) const { return _derive(x); }

		private:
			std::function<void(Matrix&)> _activate;
			std::function<Matrix(Matrix const&)> _derive;

		protected:
			ActivationFunction(
				std::function<void(Matrix&)> activate_,
				std::function<Matrix(Matrix const&)> derive_)
				: _activate(activate_), _derive(derive_)
			{}
		};

		class ErrorFunction
		{
		public:
			ErrorFunction() : _error(nullptr), _derive(nullptr) {}
			virtual float operator()(Matrix const& predicted, Matrix const& expected) const { return _error(predicted, expected); }
			virtual Matrix derive(Matrix const& predicted, Matrix const& expected) const { return _derive(predicted, expected); }

		private:
			std::function<float(Matrix const& predicted, Matrix const& expected)> _error;
			std::function<Matrix(Matrix const& predicted, Matrix const& expected)> _derive;

		protected:
			ErrorFunction(
				std::function<float(Matrix const& predicted, Matrix const& expected)> error,
				std::function<Matrix(Matrix const& predicted, Matrix const& expected)> derive)
				: _error(error), _derive(derive)
			{}
		};

		class ReLU : public ActivationFunction
		{
		public:
			ReLU() : ActivationFunction(
				[](Matrix& x) { x.map([](float x) { return std::max(0.0f, x); }); },
				[](Matrix const& x) { return x.mapped([](float v) { return v > 0 ? 1.0f : 0.0f; }); })
			{}
		};

		class Sigmoid : public ActivationFunction
		{
		public:
			Sigmoid() : ActivationFunction(
				[this](Matrix& x) { x.map([this](float x) { return _sigmoid(x); }); },
				[this](Matrix const& x) { return x.mapped([this](float x) { return _sigmoid(x) * (1.0f - _sigmoid(x)); }); })
			{}

		private:
			float _sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
		};

		class TanH : public ActivationFunction
		{
		public:
			TanH() : ActivationFunction(
				[](Matrix& x) { x.map([](float x) { return tanhf(x); }); },
				[](Matrix const& x) { return x.mapped([](float x) { float th = tanhf(x); return 1 - th * th; }); })
			{}
		};

		// https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
		// https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
		class SoftMax : public ActivationFunction
		{
		public:
			SoftMax() : ActivationFunction(
				[this](Matrix& x) { stableSoftmax(x); },
				[this](Matrix const& x)
			{
				size_t rows = x.getRowCount();
				size_t cols = x.getColCount();
				Matrix result(rows, cols);

				// Derivative of softmax uses softmax
				Matrix xSoftmax = Matrix(x);
				stableSoftmax(xSoftmax);
				for (size_t row = 0; row < rows; ++row)
				{
					for (size_t col = 0; col < cols; ++col)
					{
						// This is just the diagonal of the jacobian we need to include the off diagonals
						float softmaxVal = xSoftmax(row, col);
						result(row, col) = softmaxVal * (1.0f - softmaxVal);
					}
				}

				return result;
			})
			{}

		private:
			void stableSoftmax(Matrix& x)
			{
				size_t rows = x.getRowCount();
				size_t cols = x.getColCount();

				// Consider each row independantly
				for (size_t row = 0; row < rows; row++)
				{
					// Calculate max for stability
					float max = x(row, 0);
					for (size_t col = 1; col < cols; col++) max = std::max(max, x(row, col));

					// Apply stable SoftMax
					// Sⱼ = eᵃʲ⁺ᴰ / Σ(Cᵃᵏ⁺ᴰ)
					float sum = 0.0;
					for (size_t col = 0; col < cols; col++)
					{
						x(row, col) = std::exp(x(row, col) - max);
						sum += x(row, col);
					}
					for (size_t col = 0; col < cols; col++) x(row, col) /= sum;
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
				for (size_t i = 0; i < predicted.getRowCount(); i++)
				{
					for (size_t j = 0; j < predicted.getColCount(); j++)
					{
						float diff = expected(i, j) - predicted(i, j);
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
				// TODO: Check
				size_t rows = expected.getRowCount();
				size_t cols = expected.getColCount();

				float error = 0.0;
				for (size_t row = 0; row < rows; ++row)
				{
					for (size_t col = 0; col < cols; ++col)
					{
						error += -expected(row, col) * std::log(predicted(row, col) + float(1e-15));
					}
				}

				return error / rows;
			},
				[](Matrix const& predicted, Matrix const& expected)
			{
				// TODO: Check
				size_t rows = expected.getRowCount();
				size_t cols = expected.getColCount();

				Matrix grad(rows, cols);
				for (size_t row = 0; row < rows; row++)
				{
					for (size_t col = 0; col < cols; col++)
					{
						grad(row, col) = -(expected(row, col) / (predicted(row, col) + 1e-15f));
					}
				}

				return grad;
			})
			{}
		};
	};
}
