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

		class SoftMax : public ActivationFunction
		{
		public:
			SoftMax() : ActivationFunction(
				[](Matrix& x)
			{
				// Apply SoftMax to each row independently
				for (size_t i = 0; i < x.getRowCount(); ++i)
				{
					// Find maximum value in the row for numerical stability
					float maxVal = x(i, 0);
					for (size_t j = 1; j < x.getColCount(); ++j)
					{
						maxVal = std::max(maxVal, x(i, j));
					}

					// Apply SoftMax to each element in the row
					float sumExp = 0.0;
					for (size_t j = 0; j < x.getColCount(); ++j)
					{
						x(i, j) = std::exp(x(i, j) - maxVal);
						sumExp += x(i, j);
					}

					// Normalize the row
					for (size_t j = 0; j < x.getColCount(); ++j)
					{
						x(i, j) /= sumExp;
					}
				}
			},

				[this](Matrix const& x)
			{
				// SoftMax derivative is calculated differently than for other activation functions
				// The derivative with respect to an input xi in the SoftMax layer for class k is:
				// dSoftMax(xi)/dxi = SoftMax(xi) * (1 - SoftMax(xi))   (for i == k)
				// dSoftMax(xi)/dxi = -SoftMax(xi) * SoftMax(xk)         (for i != k)

				size_t rows = x.getRowCount();
				size_t cols = x.getColCount();

				Matrix result(rows, cols);

				// TODO: I think this is causing issues and producing inf's
				for (size_t i = 0; i < rows; ++i)
				{
					for (size_t j = 0; j < cols; ++j)
					{
						float softMax_i = std::exp(x(i, j)) / getSumExp(x, i);
						result(i, j) = -softMax_i * getSoftMaxForClass(x, i, j);
					}
				}

				return result;
			})
			{}

		private:
			float getSumExp(const Matrix& x, size_t rowIndex) const
			{
				float sumExp = 0.0;
				for (size_t j = 0; j < x.getColCount(); ++j)
				{
					sumExp += std::exp(x(rowIndex, j));
				}
				return sumExp;
			}

			float getSoftMaxForClass(const Matrix& x, size_t rowIndex, size_t classIndex) const
			{
				return (classIndex == rowIndex) ? (1.0f - std::exp(x(rowIndex, classIndex)) / getSumExp(x, rowIndex))
					: (-std::exp(x(rowIndex, classIndex)) / getSumExp(x, rowIndex));
			}
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
				for (size_t i = 0; i < rows; ++i)
				{
					for (size_t j = 0; j < cols; ++j)
					{
						error += expected(i, j) * std::log(predicted(i, j) + 1e-15f);
					}
				}
				return -error / rows;
			},

				[](Matrix const& predicted, Matrix const& expected)
			{
				size_t rows = expected.getRowCount();
				size_t cols = expected.getColCount();

				Matrix grad(rows, cols);
				for (size_t i = 0; i < rows; ++i)
				{
					for (size_t j = 0; j < cols; ++j)
					{
						grad(i, j) = -expected(i, j) / (predicted(i, j) + 1e-15f);
					}
				}
				return grad;
			})
			{}
		};
	};
}
