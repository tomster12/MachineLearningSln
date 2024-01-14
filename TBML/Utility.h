#pragma once

#include "stdafx.h"
#include "Matrix.h"

namespace tbml
{
	namespace fn
	{
		float getRandomFloat();

		class ActivationFunction
		{
		public:
			ActivationFunction() : _activate(nullptr), _partialDerivative(nullptr) {}
			virtual void operator()(Matrix& x) const { _activate(x); }
			virtual Matrix partialDerivative(Matrix const& x, Matrix const& pdNeuronOut) const { return _partialDerivative(x, pdNeuronOut); }

		private:
			std::function<void(Matrix&)> _activate;
			std::function<Matrix(Matrix const&, Matrix const&)> _partialDerivative;

		protected:
			ActivationFunction(
				std::function<void(Matrix&)> activate_,
				std::function<Matrix(Matrix const&, Matrix const&)> partialDerivative_)
				: _activate(activate_), _partialDerivative(partialDerivative_)
			{}
		};

		class ErrorFunction
		{
		public:
			ErrorFunction() : _error(nullptr), _partialDerivative(nullptr) {}
			virtual float operator()(Matrix const& predicted, Matrix const& expected) const { return _error(predicted, expected); }
			virtual Matrix partialDerivative(Matrix const& predicted, Matrix const& expected) const { return _partialDerivative(predicted, expected); }

		private:
			std::function<float(Matrix const& predicted, Matrix const& expected)> _error;
			std::function<Matrix(Matrix const& predicted, Matrix const& expected)> _partialDerivative;

		protected:
			ErrorFunction(
				std::function<float(Matrix const& predicted, Matrix const& expected)> error,
				std::function<Matrix(Matrix const& predicted, Matrix const& expected)> partialDerivative)
				: _error(error), _partialDerivative(partialDerivative)
			{}
		};

		class ReLU : public ActivationFunction
		{
		public:
			ReLU() : ActivationFunction(
				[](Matrix& x) { x.map([](float x) { return std::max(0.0f, x); }); },
				[](Matrix const& x, Matrix const& pdNeuronOut) { return x.mapped([](float v) { return v > 0 ? 1.0f : 0.0f; }) * pdNeuronOut; })
			{}
		}; //nIn[layer] *= pdToOut;

		class Sigmoid : public ActivationFunction
		{
		public:
			Sigmoid() : ActivationFunction(
				[this](Matrix& x) { x.map([this](float x) { return _sigmoid(x); }); },
				[this](Matrix const& x, Matrix const& pdNeuronOut) { return x.mapped([this](float x) { return _sigmoid(x) * (1.0f - _sigmoid(x)); }) * pdNeuronOut; })
			{}

		private:
			float _sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
		};

		class TanH : public ActivationFunction
		{
		public:
			TanH() : ActivationFunction(
				[](Matrix& x) { x.map([](float x) { return tanhf(x); }); },
				[](Matrix const& x, Matrix const& pdNeuronOut) { return x.mapped([](float x) { float th = tanhf(x); return 1 - th * th; }) * pdNeuronOut; })
			{}
		};

		// https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
		// https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
		// https://www.v7labs.com/blog/cross-entropy-loss-guide#h3
		// https://stats.stackexchange.com/questions/277203/differentiation-of-cross-entropy
		// https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/

		class SoftMax : public ActivationFunction
		{
		public:
			SoftMax() : ActivationFunction(
				[this](Matrix& x) { stableSoftmax(x); },
				[this](Matrix const& s, Matrix const& pdNeuronOut)
			{
				size_t rows = s.getRowCount();
				size_t cols = s.getColCount();
				Matrix result(rows, cols);

				for (size_t row = 0; row < rows; row++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						float Sj = s(row, j);
						for (size_t i = 0; i < cols; i++)
						{
							float Si = s(row, i);
							int Dij = (j == i) ? 1 : 0;
							result(row, j) += (Si * (Dij - Sj)) * pdNeuronOut(row, i);
						}
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
				size_t rows = expected.getRowCount();
				size_t cols = expected.getColCount();

				float error = 0.0;
				for (size_t row = 0; row < rows; ++row)
				{
					for (size_t col = 0; col < cols; col++)
					{
						// error += -expected(row, col) * std::log(predicted(row, col));
						error += -expected(row, col) * std::log(predicted(row, col) + float(1e-15));
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
					// TODO: This seems to cause issues and get out of control but surely theres a good way
					// I have an inkling that nan happens when its going well but who knows
					for (size_t col = 0; col < cols; col++)
					{
						grad(row, col) = -expected(row, col) / (predicted(row, col) + 1e-15f);
						// grad(row, col) = std::min(std::max(-expected(row, col) / (predicted(row, col) + 1e-15f), -5000.0f), 5000.0f);
					}
				}

				return grad;
			})
			{}
		};
	};
}
