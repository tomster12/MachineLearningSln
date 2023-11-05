
#include "stdafx.h"
#include "Matrix.h"

namespace tbml
{
	Matrix::Matrix()
	{
		this->data = std::vector<float>();
	}

	Matrix::Matrix(const std::vector<std::vector<float>>& data)
	{
		this->rows = data.size();
		this->cols = (rows > 0) ? data[0].size() : 0;
		this->data = std::vector<float>(rows * cols);
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				this->data[i * cols + j] = data[i][j];
			}
		}
	}

	Matrix::Matrix(std::vector<float>&& data, size_t rows, size_t cols)
	{
		this->rows = rows;
		this->cols = cols;
		this->data = std::move(data);
	}

	Matrix::Matrix(size_t rows, size_t cols)
	{
		resize(rows, cols);
	}

	void Matrix::resize(size_t rows, size_t cols)
	{
		this->rows = rows;
		this->cols = cols;
		data = std::vector<float>(rows * cols);
		for (size_t i = 0; i < rows * cols; i++) data[i] = 0.0f;
	}

	void Matrix::clear()
	{
		rows = 0;
		cols = 0;
		data.clear();
	}

	Matrix& Matrix::map(std::function<float(float)> func)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] = func(data[i * cols + j]);
			}
		}
		return *this;
	}

	Matrix& Matrix::ewise(Matrix const& m, std::function<float(float, float)> func)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] = func(data[i * cols + j], m.data[(i % m.rows) * m.cols + (j % m.cols)]);
			}
		}
		return *this;
	}

	Matrix& Matrix::transpose()
	{
		std::vector<float> newData(rows * cols);
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				newData[j * rows + i] = data[i * cols + j];
			}
		}
		data = std::move(newData);
		size_t tmp = rows;
		rows = cols;
		cols = tmp;
		return *this;
	}

	Matrix& Matrix::cross(Matrix const& m)
	{
		// Massive speedup but cannot parrellelise or const
		// static std::vector<float> calcRow(oCols);
		//if (calcRow.size() != oCols) calcRow.resize(oCols);

		std::vector<float> newData(rows * m.cols);

		for (size_t i = 0; i < rows; i++)
		{
			for (size_t mj = 0; mj < m.cols; mj++)
			{
				float sum = 0;
				for (size_t j = 0; j < cols; j++) sum += data[i * cols + j] * m.data[j * m.cols + mj];
				newData[i * m.cols + mj] = sum;
			}
		}

		data = std::move(newData);
		cols = m.cols;
		return *this;
	}

	float Matrix::acc(std::function<float(float, float)> func, float initial) const
	{
		float current = initial;
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				current = func(data[i * cols + j], current);
			}
		}
		return current;
	}

	void Matrix::printValues(std::string tag) const
	{
		std::cout << tag << std::endl;
		for (size_t i = 0; i < rows; i++)
		{
			std::cout << "  ";
			for (size_t j = 0; j < cols; j++)
			{
				char prefix = (data[i * cols + j] >= 0) ? ' ' : '\0';
				std::cout << prefix << std::fixed << std::setprecision(4) << data[i * cols + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	void Matrix::printDims(std::string tag) const
	{
		std::cout << tag << rows << " x " << cols << std::endl;
	}

	std::vector<Matrix> Matrix::getSplitRows(size_t splitSize) const
	{
		size_t splitCount = (size_t)(ceil((float)rows / splitSize));
		std::vector<Matrix> splits = std::vector<Matrix>(splitCount);

		for (size_t si = 0; si < splitCount; si++)
		{
			size_t splitRows = (si < splitCount - 1) ? splitSize : (rows % splitSize);
			std::vector<float> splitData = std::vector<float>(splitRows * cols);

			for (size_t i = 0; i < splitRows; i++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					splitData[i * cols + j] = data[(si * splitSize + i) * cols + j];

				}
			}

			splits[si] = Matrix(std::move(splitData), splitRows, cols);
		}

		return splits;
	}
}
