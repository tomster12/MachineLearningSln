#pragma once

#include <cassert>
#include "Tensor.h"

namespace tbml
{
	namespace fn
	{
		float getRandomFloat();

		float classificationAccuracy(const tbml::Tensor& output, const tbml::Tensor& expected);

		class ActivationFunction
		{
		public:
			ActivationFunction() : activateFn(nullptr), chainDerivativeFn(nullptr) {}
			virtual void activate(Tensor& x) const { activateFn(x); }
			virtual Tensor chainDerivative(const Tensor& z, const Tensor& pdToOut) const { return chainDerivativeFn(z, pdToOut); }

		private:
			std::function<void(Tensor&)> activateFn;
			std::function<Tensor(const Tensor&, const Tensor&)> chainDerivativeFn;

		protected:
			ActivationFunction(
				std::function<void(Tensor& x)> activateFn,
				std::function<Tensor(const Tensor& z, const Tensor& pdToOut)> chainDerivativeFn)
				: activateFn(activateFn), chainDerivativeFn(chainDerivativeFn)
			{}
		};

		class LossFunction
		{
		public:
			LossFunction() : lossFn(nullptr), derivativeFn(nullptr) {}
			virtual float activate(const Tensor& output, const Tensor& expected) const { return lossFn(output, expected); }
			virtual Tensor derive(const Tensor& output, const Tensor& expected) const { return derivativeFn(output, expected); }

		private:
			std::function<float(const Tensor& output, const Tensor& expected)> lossFn;
			std::function<Tensor(const Tensor& output, const Tensor& expected)> derivativeFn;

		protected:
			LossFunction(
				std::function<float(const Tensor& output, const Tensor& expected)> lossFn,
				std::function<Tensor(const Tensor& output, const Tensor& expected)> derivativeFn)
				: lossFn(lossFn), derivativeFn(derivativeFn)
			{}
		};

		class ReLU : public ActivationFunction
		{
		public:
			ReLU() : ActivationFunction(
				[](Tensor& x)
			{
				x.map([](float x) { return std::max(0.0f, x); });
			},
				[](const Tensor& z, const Tensor& pdToOut)
			{
				Tensor pdOutToIn = z.mapped([](float v) { return v > 0 ? 1.0f : 0.0f; });
				return pdOutToIn * pdToOut;
			})
			{}
		};

		class Sigmoid : public ActivationFunction
		{
		public:
			Sigmoid() : ActivationFunction(
				[this](Tensor& x)
			{
				x.map([this](float x) { return sigmoid(x); });
			},
				[this](const Tensor& z, const Tensor& pdToOut)
			{
				Tensor pdOutToIn = z.mapped([this](float v)
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

		class TanH : public ActivationFunction
		{
		public:
			TanH() : ActivationFunction(
				[](Tensor& x)
			{
				x.map([](float x) { return tanhf(x); });
			},
				[](const Tensor& z, const Tensor& pdToOut)
			{
				Tensor pdOutToIn = z.mapped([](float v)
				{
					float th = tanhf(v);
					return 1 - th * th;
				});

				return pdOutToIn * pdToOut;
			})
			{}
		};

		class SoftMax : public ActivationFunction
		{
		public:
			SoftMax() : ActivationFunction(
				[this](Tensor& x)
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
				[this](const Tensor& z, const Tensor& pdToOut)
			{
				auto shape = z.getShape();
				assert(shape.size() == 2);
				Tensor result(shape, 0);

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

		class SquareError : public LossFunction
		{
		public:
			SquareError() : LossFunction(
				[](const Tensor& output, const Tensor& expected)
			{
				const auto& predicteddata = output.getData();
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
				[](const Tensor& output, const Tensor& expected)
			{
				assert(output.getShape() == expected.getShape());

				// derivative of square error = YH - Y
				return output - expected;
			})
			{}
		};

		class CrossEntropy : public LossFunction
		{
		public:
			CrossEntropy() : LossFunction(
				[](const Tensor& output, const Tensor& expected)
			{
				const auto& predictedData = output.getData();
				const auto& expectedData = expected.getData();
				assert(predictedData.size() == expectedData.size());

				// Cross entropy = Σ -Yi * log(YHi + e) with epsilon = 1e-15f for stability
				float error = 0.0;
				for (size_t i = 0; i < predictedData.size(); i++)
				{
					error += -expectedData[i] * std::log(predictedData[i] + float(1e-15f));
				}
				return error / output.getShape()[0];
			},
				[](const Tensor& output, const Tensor& expected)
			{
				assert(output.getShape() == expected.getShape());

				// derivative of cross entropy = Yi / (YHi + e) with epsilon = 1e-15f for stability
				return expected.ewised(output, [](float expected, float output) { return -expected / (output + 1e-15f); });
			})
			{}
		};
	};
}
