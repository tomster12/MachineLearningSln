
#include "stdafx.h"
#include "Matrix.h"

namespace tbml
{
#pragma region Setup

	Matrix::Matrix()
	{
		// Initialize variables
		data = std::vector<std::vector<float>>();
		rows = 0;
		cols = 0;
	}

	Matrix::Matrix(std::vector<std::vector<float>> data)
	{
		// Initialize variables
		this->data = data;
		this->rows = data.size();
		this->cols = (rows > 0) ? data[0].size() : 0;
	}

	Matrix::Matrix(size_t rows, size_t cols)
	{
		// Initialize variables
		resize(rows, cols);
	}

	void Matrix::resize(size_t rows, size_t cols)
	{
		// Set to a specific size and zero
		this->rows = rows;
		this->cols = cols;
		data = std::vector<std::vector<float>>(rows);
		for (size_t row = 0; row < rows; row++)
		{
			data[row] = std::vector<float>(cols);
			for (size_t col = 0; col < cols; col++) data[row][col] = 0.0f;
		}
	}

	void Matrix::clear()
	{
		// Clear out matrix
		data.clear();
		rows = 0;
		cols = 0;
	}

#pragma endregion

#pragma region Arithmetic

	Matrix Matrix::cross(Matrix& other)
	{
		// Check compatibility
		size_t oRows = other.getRows();
		size_t oCols = other.getCols();
		std::vector<std::vector<float>>& oData = other.getData();
		if (cols != oRows) throw std::invalid_argument("Matrix dimensions do not match.");

		// Create new matrix and inplace cross with other
		Matrix newMatrix = *this;
		newMatrix.icross(other);
		return newMatrix;
	}

	Matrix* Matrix::icross(Matrix& other)
	{
		// Check compatibility
		size_t oRows = other.getRows();
		size_t oCols = other.getCols();
		std::vector<std::vector<float>>& oData = other.getData();
		if (cols != oRows) throw std::invalid_argument("Matrix dimensions do not match.");

		// Inplace cross with other
		for (size_t row = 0; row < rows; row++)
		{
			std::vector<float> newRow(oCols);
			for (size_t oCol = 0; oCol < oCols; oCol++)
			{
				float sum = 0;
				for (size_t i = 0; i < cols; i++) sum += data[row][i] * oData[i][oCol];
				newRow[oCol] = sum;
			}
			data[row] = newRow;
		}

		// Update variables and return
		rows = rows;
		cols = oCols;
		return this;
	}

	Matrix Matrix::transpose()
	{
		// Create new matrix and inplace tranpose
		Matrix newMatrix = *this;
		newMatrix.itranspose();
		return newMatrix;
	}

	Matrix* Matrix::itranspose()
	{
		// Inplace tranpose this
		std::vector<std::vector<float>> newData(cols);
		for (size_t row = 0; row < cols; row++)
		{
			std::vector<float> newRow(rows);
			for (size_t col = 0; col < rows; col++)
			{
				newRow[col] = data[col][row];
			}
			newData[row] = newRow;
		}

		// Update variables and return
		data = newData;
		size_t tmp = rows;
		rows = cols;
		cols = tmp;
		return this;
	}

	float Matrix::acc(float (*func)(float, float), float initial)
	{
		// Apply function to each element in matrix
		float current = initial;
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				current = func(data[row][col], current);
			}
		}
		return current;
	}

	Matrix Matrix::map(float (*func)(float))
	{
		// Create new matrix and inplace map
		Matrix newMatrix = *this;
		newMatrix.imap(func);
		return newMatrix;
	}

	Matrix* Matrix::imap(float (*func)(float))
	{
		// Apply function to each element
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row][col] = func(data[row][col]);
			}
		}
		return this;
	}

	Matrix Matrix::ewise(Matrix& other, float (*func)(float, float))
	{
		// Create new matrix and inplace map
		Matrix newMatrix = *this;
		newMatrix.iewise(other, func);
		return newMatrix;
	}

	Matrix* Matrix::iewise(Matrix& other, float (*func)(float, float))
	{
		// Get variables
		size_t oRows = other.getRows();
		size_t oCols = other.getCols();
		std::vector<std::vector<float>>& oData = other.getData();

		// Inplace add to other matrix
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				float value = data[row][col];
				float oValue = oData[row < oRows ? row : oRows - 1][col < oCols ? col : oCols - 1];
				data[row][col] = func(value, oValue);
			}
		}
		return this;
	}

	Matrix Matrix::scale(float val)
	{
		// Create new matrix and inplace scale
		Matrix newMatrix = *this;
		newMatrix.iscale(val);
		return newMatrix;
	}

	Matrix* Matrix::iscale(float val)
	{
		// Apply function to matrix
		for (size_t row = 0; row < rows; row++)
		{
			for (size_t col = 0; col < cols; col++)
			{
				data[row][col] = data[row][col] * val;
			}
		}
		return this;
	}

	Matrix Matrix::sub(Matrix& other) { return ewise(other, [](float x, float y) { return x - y; }); }

	Matrix* Matrix::isub(Matrix& other) { return iewise(other, [](float x, float y) { return x - y; }); }

	Matrix Matrix::add(Matrix& other) { return ewise(other, [](float x, float y) { return x + y; }); }

	Matrix* Matrix::iadd(Matrix& other) { return iewise(other, [](float x, float y) { return x + y; }); }

	Matrix Matrix::times(Matrix& other) { return ewise(other, [](float x, float y) { return x * y; }); }

	Matrix* Matrix::itimes(Matrix& other) { return iewise(other, [](float x, float y) { return x * y; }); }

	Matrix Matrix::div(Matrix& other) { return ewise(other, [](float x, float y) { return x / y; }); }

	Matrix* Matrix::idiv(Matrix& other) { return iewise(other, [](float x, float y) { return x / y; }); }

#pragma endregion

#pragma region Main

	void Matrix::printValues(std::string tag)
	{
		// Print each layers values
		std::cout << tag << std::endl;
		for (size_t row = 0; row < rows; row++)
		{
			std::cout << "  ";
			for (size_t col = 0; col < cols; col++)
			{
				char prefix = (data[row][col] >= 0) ? ' ' : '\0';
				std::cout << prefix << std::fixed << std::setprecision(4) << data[row][col] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	void Matrix::printDims(std::string tag)
	{
		// Print dimensions
		std::cout << tag << rows << " x " << cols << std::endl;
	}

	std::vector<Matrix> Matrix::splitRows(size_t splitSize) const
	{
		// Initialize variables
		size_t splitCount = (int)(rows / splitSize);
		std::vector<Matrix> splits = std::vector<Matrix>(splitCount + ((rows % splitSize == 0) ? 0 : 1));

		// Loop over each split and copy rows
		for (size_t i = 0; i < splitCount; i++)
		{
			std::vector<std::vector<float>> splitData = std::vector<std::vector<float>>(splitSize);
			for (size_t o = 0; o < splitSize; o++) splitData[o] = data[i * splitSize + o];
			splits[i] = Matrix(splitData);
		}

		// Copy rows for last split
		if ((rows % splitSize) != 0)
		{
			std::vector<std::vector<float>> splitData = std::vector<std::vector<float>>(rows % splitSize);
			for (size_t o = 0; o < rows % splitSize; o++) splitData[o] = data[splitCount * splitSize + o];
			splits[splitCount] = Matrix(splitData);
		}

		// Return splits
		return splits;
	}

	std::vector<std::vector<float>>& Matrix::getData() { return data; }

	size_t Matrix::getRows() const { return rows; }

	size_t Matrix::getCols() const { return cols; }

	float Matrix::get(size_t row, size_t col) const
	{
		// Return data, and check if indices are inbound
		if (row >= 0 && col >= 0 && row < rows && col < cols) return data[row][col];
		else throw std::invalid_argument("Index out of range.");
	}

	void Matrix::set(size_t row, size_t col, float val)
	{
		// Set data, and check if indices are inbound
		if (row >= 0 && col >= 0 && row < rows && col < cols) data[row][col] = val;
		else throw std::invalid_argument("Index out of range.");
	}

	bool Matrix::getEmpty() const { return rows == 0 || cols == 0; }

#pragma endregion
}
