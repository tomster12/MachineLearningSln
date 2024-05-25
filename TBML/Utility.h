#pragma once

#include <cassert>
#include "Tensor.h"

namespace tbml
{
	namespace fn
	{
		float getRandomFloat();

		int getRandomInt(int min, int max);

		size_t argmax(const tbml::Tensor& tensor, size_t row);

		float classificationAccuracy(const tbml::Tensor& output, const tbml::Tensor& expected);

		class SquareError;
		class CrossEntropy;

		class LossFunction
		{
		public:
			virtual float calculate(const Tensor& output, const Tensor& expected) const = 0;
			virtual Tensor derivative(const Tensor& output, const Tensor& expected) const = 0;
			virtual void serialize(std::ostream& os) const = 0;
			static std::shared_ptr<LossFunction> deserialize(std::istream& is);
		};

		using LossFunctionPtr = std::shared_ptr<tbml::fn::LossFunction>;

		class SquareError : public LossFunction
		{
		public:
			float calculate(const Tensor& output, const Tensor& expected) const override
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

			Tensor derivative(const Tensor& output, const Tensor& expected) const override
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
			float calculate(const Tensor& output, const Tensor& expected) const override
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

			Tensor derivative(const Tensor& output, const Tensor& expected) const override
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
