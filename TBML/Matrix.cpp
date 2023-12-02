
#include "stdafx.h"
#include "Matrix.h"
#include <omp.h>

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

	Matrix& Matrix::addBounded(Matrix const& m)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] = data[i * cols + j] + m.data[std::min(i, m.rows - 1) * m.cols + std::min(j, m.cols - 1)];
			}
		}
		return *this;
	}

	Matrix& Matrix::operator+=(Matrix const& m)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] += m.data[i * cols + j];
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
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] -= m.data[i * cols + j];
			}
		}
		return *this;
	}

	Matrix& Matrix::operator-=(float v)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] -= v;
			}
		}
		return *this;
	}

	Matrix& Matrix::operator*=(Matrix const& m)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] *= m.data[i * cols + j];
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
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] /= m.data[i * cols + j];
			}
		}
		return *this;
	}

	Matrix& Matrix::operator/=(float v)
	{
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				data[i * cols + j] /= v;
			}
		}
		return *this;
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
				data[i * cols + j] = func(data[i * cols + j], m.data[i * m.cols + j]);
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
		const std::vector<float>& a = data;
		const std::vector<float>& b = m.data;
		std::vector<float> c(rows * m.cols);

		//#pragma omp parallel for num_threads(4)
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t mj = 0; mj < m.cols; mj++)
			{
				for (size_t j = 0; j < cols; j++)
				{
					c[i * m.cols + mj] += a[i * cols + j] * b[j * m.cols + mj];
				}
			}
		}

		data = std::move(c);
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
