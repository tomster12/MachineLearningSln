#pragma once

#include <cassert>
#include "Tensor.h"

namespace tbml
{
	namespace fn
	{
		float getRandomFloat();

		float classificationAccuracy(const tbml::Tensor& output, const tbml::Tensor& expected);

		class ActivationFunction;
		class ReLU;
		class Sigmoid;
		class TanH;
		class SoftMax;

		class ActivationFunction
		{
		public:
			virtual void activate(Tensor& x) const = 0;
			virtual Tensor chainDerivative(const Tensor& z, const Tensor& pdToOut) const = 0;

			virtual void serialize(std::ostream& os) const = 0;
			static std::shared_ptr<ActivationFunction> deserialize(std::istream& is);
		};

		class SquareError;
		class CrossEntropy;

		class LossFunction
		{
		public:
			virtual float activate(const Tensor& output, const Tensor& expected) const = 0;
			virtual Tensor derive(const Tensor& output, const Tensor& expected) const = 0;

			virtual void serialize(std::ostream& os) const = 0;
			static std::shared_ptr<LossFunction> deserialize(std::istream& is);
		};

		using ActivationFunctionPtr = std::shared_ptr<fn::ActivationFunction>;
		using LossFunctionPtr = std::shared_ptr<fn::LossFunction>;

		// Output range [0, inf]
		class ReLU : public ActivationFunction
		{
		public:
			void activate(Tensor& x) const override
			{
				x.map([](float x) { return std::max(0.0f, x); });
			};

			Tensor chainDerivative(const Tensor& z, const Tensor& pdToOut) const override
			{
				Tensor pdOutToIn = z.mapped([](float v) { return v > 0 ? 1.0f : 0.0f; });
				return pdOutToIn * pdToOut;
			};

			void serialize(std::ostream& os) const override
			{
				os << "ReLU\n";
			}
		};

		// Output range [0, 1]
		class Sigmoid : public ActivationFunction
		{
		public:
			void activate(Tensor& x) const override
			{
				x.map([this](float x) { return sigmoid(x); });
			};

			Tensor chainDerivative(const Tensor& z, const Tensor& pdToOut) const override
			{
				Tensor pdOutToIn = z.mapped([this](float v)
				{
					float sv = sigmoid(v);
					return sv * (1.0f - sv);
				});

				return pdOutToIn * pdToOut;
			};

			void serialize(std::ostream& os) const override
			{
				os << "Sigmoid\n";
			}

		private:
			float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
		};

		// Output range [-1, 1]
		class TanH : public ActivationFunction
		{
		public:
			void activate(Tensor& x) const override
			{
				x.map([](float x) { return tanhf(x); });
			};

			Tensor chainDerivative(const Tensor& z, const Tensor& pdToOut) const override
			{
				Tensor pdOutToIn = z.mapped([](float v)
				{
					float th = tanhf(v);
					return 1 - th * th;
				});

				return pdOutToIn * pdToOut;
			};

			void serialize(std::ostream& os) const override
			{
				os << "TanH\n";
			}
		};

		// Output range [0, 1] per row, sum of row = 1
		class SoftMax : public ActivationFunction
		{
		public:
			void activate(Tensor& x) const override
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
			};

			Tensor chainDerivative(const Tensor& z, const Tensor& pdToOut) const override
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
			};

			void serialize(std::ostream& os) const override
			{
				os << "SoftMax\n";
			}
		};

		class SquareError : public LossFunction
		{
		public:
			float activate(const Tensor& output, const Tensor& expected) const override
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
			};

			Tensor derive(const Tensor& output, const Tensor& expected) const override
			{
				assert(output.getShape() == expected.getShape());

				// derivative of square error = YH - Y
				return output - expected;
			};

			void serialize(std::ostream& os) const override
			{
				os << "SquareError\n";
			}
		};

		class CrossEntropy : public LossFunction
		{
		public:
			float activate(const Tensor& output, const Tensor& expected) const override
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
			};

			Tensor derive(const Tensor& output, const Tensor& expected) const override
			{
				assert(output.getShape() == expected.getShape());

				// derivative of cross entropy = Yi / (YHi + e) with epsilon = 1e-15f for stability
				return expected.ewised(output, [](float expected, float output) { return -expected / (output + 1e-15f); });
			};

			void serialize(std::ostream& os) const override
			{
				os << "CrossEntropy\n";
			}
		};
	};
}
