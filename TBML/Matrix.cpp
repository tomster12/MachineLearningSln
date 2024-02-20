#include "stdafx.h"
#include "Matrix.h"
#include <omp.h>

namespace tbml
{
	Matrix::Matrix()
	{
		this->data = std::vector<float>();
	}

	Matrix::Matrix(const Matrix& m)
	{
		this->rows = m.getRowCount();
		this->cols = m.getColCount();
		this->data = std::vector<float>(rows * cols);

		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				this->data[row * cols + col] = m(row, col);
			}
		}
	}

	Matrix::Matrix(const std::vector<std::vector<float>>& data)
	{
		this->rows = data.size();
		this->cols = (rows > 0) ? data[0].size() : 0;
		this->data = std::vector<float>(rows * cols);

		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				this->data[row * cols + col] = data[row][col];
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

	Matrix& Matrix::operator+=(Matrix const& m)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[row * cols + j] += m.data[row * cols + j];
			}
		}

		return *this;
	}

	Matrix& Matrix::operator+=(float v)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] += v;
			}
		}

		return *this;
	}

	Matrix& Matrix::operator-=(Matrix const& m)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row * cols + col] -= m.data[row * cols + col];
			}
		}

		return *this;
	}

	Matrix& Matrix::operator-=(float v)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row * cols + col] -= v;
			}
		}

		return *this;
	}

	Matrix& Matrix::operator*=(Matrix const& m)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row * cols + col] *= m.data[row * cols + col];
			}
		}

		return *this;
	}

	Matrix& Matrix::operator*=(float v)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] *= v;
			}
		}

		return *this;
	}

	Matrix& Matrix::operator/=(Matrix const& m)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row * cols + col] /= m.data[row * cols + col];
			}
		}

		return *this;
	}

	Matrix& Matrix::operator/=(float v)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row * cols + col] /= v;
			}
		}

		return *this;
	}

	Matrix& Matrix::map(std::function<float(float)> func)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row * cols + col] = func(data[row * cols + col]);
			}
		}

		return *this;
	}

	Matrix& Matrix::ewise(Matrix const& m, std::function<float(float, float)> func)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row * cols + col] = func(data[row * cols + col], m.data[row * m.cols + col]);
			}
		}

		return *this;
	}

	Matrix& Matrix::transpose()
	{
		std::vector<float> result(rows * cols);
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				result[col * rows + row] = data[row * cols + col];
			}
		}

		data = std::move(result);
		size_t tmp = rows;
		rows = cols;
		cols = tmp;
		return *this;
	}

	Matrix& Matrix::cross(Matrix const& m)
	{
		const std::vector<float>& a = data;
		const std::vector<float>& b = m.data;
		std::vector<float> result(rows * m.cols);

		#pragma omp parallel for num_threads(4)
		for (int row = 0; row < (int)rows; row++)
		{
			for (int mcol = 0; mcol < (int)m.cols; mcol++)
			{
				for (int col = 0; col < (int)cols; col++)
				{
					result[row * m.cols + mcol] += a[row * cols + col] * b[col * m.cols + mcol];
				}
			}
		}

		data = std::move(result);
		cols = m.cols;
		return *this;
	}

	float Matrix::acc(std::function<float(float, float)> func, float initial) const
	{
		float current = initial;
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				current = func(data[row * cols + col], current);
			}
		}

		return current;
	}

	Matrix& Matrix::addBounded(Matrix const& m)
	{
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row * cols + col] = data[row * cols + col] + m.data[std::min(row, m.rows - 1) * m.cols + std::min(col, m.cols - 1)];
			}
		}

		return *this;
	}

	void Matrix::printValues(std::string tag) const
	{
		std::cout << tag << std::endl;
		for (size_t row = 0; row < rows; row++)
		{
			std::cout << "  ";
			for (size_t col = 0; col < cols; col++)
			{
				char prefix = (data[row * cols + col] >= 0) ? ' ' : '\0';
				std::cout << prefix << std::fixed << std::setprecision(4) << data[row * cols + col] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	void Matrix::printDims(std::string tag) const
	{
		std::cout << tag << rows << " x " << cols << std::endl;
	}

	std::vector<Matrix> Matrix::groupRows(size_t targetGroupSize) const
	{
		size_t groupCount = (size_t)(ceil((float)rows / targetGroupSize));
		bool hasUneven = (rows % targetGroupSize) != 0;
		std::vector<Matrix> groups = std::vector<Matrix>(groupCount);

		for (size_t group = 0; group < groupCount; group++)
		{
			size_t groupSize = (hasUneven && (group == groupCount - 1)) ? (rows % targetGroupSize) : targetGroupSize;
			std::vector<float> groupData = std::vector<float>(groupSize * cols);

			for (size_t row = 0; row < groupSize; row++)
			{
				for (size_t col = 0; col < cols; col++)
				{
					groupData[row * cols + col] = data[(group * groupSize + row) * cols + col];
				}
			}

			groups[group] = Matrix(std::move(groupData), groupSize, cols);
		}

		return groups;
	}
}
