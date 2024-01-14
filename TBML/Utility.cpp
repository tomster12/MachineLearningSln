#include "Utility.h"

namespace tbml
{
	namespace fn
	{
		float getRandomFloat()
		{
			return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		}

		float calculateAccuracy(tbml::Matrix const& predicted, tbml::Matrix const& expected)
		{
			size_t rows = predicted.getRowCount();
			size_t cols = predicted.getColCount();

			float accuracy = 0.0f;
			for (size_t row = 0; row < rows; row++)
			{
				int predictedClass = 0;
				float predictedValue = predicted(row, 0);
				for (size_t i = 1; i < cols; i++)
				{
					float v = predicted(row, i);
					if (v > predictedValue)
					{
						predictedClass = i;
						predictedValue = v;
					}
				}

				int expectedClass = 0;
				float expectedValue = expected(row, 0);
				for (size_t i = 1; i < cols; i++)
				{
					float v = expected(row, i);
					if (v > expectedValue)
					{
						expectedClass = i;
						expectedValue = v;
					}
				}

				accuracy += ((predictedClass == expectedClass) ? 1.0f : 0.0f) / rows;
			}

			return accuracy;
		}
	}
}
