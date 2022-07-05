
#include "stdafx.h"
#include "Matrix.h"


namespace tbml {

	#pragma region Setup

	Matrix::Matrix() {
		// Initialize variables
		data = std::vector<std::vector<float>>();
		rows = 0;
		cols = 0;
	}

	Matrix::Matrix(std::vector<std::vector<float>> data_) {
		// Initialize variables
		data = data_;
		rows = data.size();
		cols = (rows > 0) ? data[0].size() : 0;
	}

	Matrix::Matrix(size_t rows_, size_t cols_) {
		// Initialize variables
		data = std::vector<std::vector<float>>();
		rows = rows_;
		cols = cols_;
		for (size_t row = 0; row < rows; row++) {
			data.push_back(std::vector<float>());
			for (size_t col = 0; col < cols; col++) {
				data[row].push_back(0);
			}
		}
	}

	#pragma endregion


	#pragma region Arithmetic

	Matrix Matrix::cross(Matrix& other) {
		// Get variables and check dimensions compatible
		std::vector<std::vector<float>> newData;
		size_t oRows = other.getRows();
		size_t oCols = other.getCols();
		std::vector<std::vector<float>>& oData = other.getData();
		if (cols != oRows) throw std::invalid_argument("Matrix dimensions do not match.");

		// Matrix multiply
		for (size_t newRow = 0; newRow < rows; newRow++) {
			newData.push_back(std::vector<float>());
			for (size_t newCol = 0; newCol < oCols; newCol++) {
				float sum = 0;
				for (size_t i = 0; i < cols; i++)
					sum += data[newRow][i] * oData[i][newCol];
				newData[newRow].push_back(sum);
			}
		}
		return Matrix(newData);
	}

	Matrix* Matrix::icross(Matrix& other) {
		// Get variables and check dimensions compatible
		size_t oRows = other.getRows();
		size_t oCols = other.getCols();
		std::vector<std::vector<float>>& oData = other.getData();
		if (cols != oRows) throw std::invalid_argument("Matrix dimensions do not match.");

		// Inplace matrix multiply
		for (size_t row = 0; row < rows; row++) {
			std::vector<float> newRow;
			for (size_t col = 0; col < oCols; col++) {
				float sum = 0;
				for (size_t i = 0; i < cols; i++)
					sum += data[row][i] * oData[i][col];
				newRow.push_back(sum);
			}
			data[row] = newRow;
		}

		// Update variables
		rows = rows;
		cols = oCols;
		return this;
	}


	Matrix Matrix::transpose() {
		// Tranpose matrix
		std::vector<std::vector<float>> newData;
		for (size_t newRow = 0; newRow < cols; newRow++) {
			newData.push_back(std::vector<float>());
			for (size_t newCol = 0; newCol < rows; newCol++) {
				newData[newRow].push_back(data[newCol][newRow]);
			}
		}
		return Matrix(newData);
	}

	Matrix* Matrix::itranspose() {
		// Inplace tranpose matrix
		data = std::vector<std::vector<float>>();
		for (size_t row = 0; row < cols; row++) {
			data.push_back(std::vector<float>());
			for (size_t col = 0; col < rows; col++) {
				data[row].push_back(data[col][row]);
			}
		}

		// Update variables
		size_t tmp = rows;
		rows = cols;
		cols = tmp;
		return this;
	}


	float Matrix::acc(float (*func)(float, float), float initial) {
		// Apply function to each element in matrix
		float current = initial;
		for (size_t row = 0; row < rows; row++) {
			for (size_t col = 0; col < cols; col++) {
				current = func(data[row][col], current);
			}
		}
		return current;
	}


	Matrix Matrix::map(float (*func)(float)) {
		// Apply function to matrix
		std::vector<std::vector<float>> newData;
		for (size_t newRow = 0; newRow < rows; newRow++) {
			newData.push_back(std::vector<float>());
			for (size_t newCol = 0; newCol < cols; newCol++) {
				newData[newRow].push_back(func(data[newRow][newCol]));
			}
		}
		return Matrix(newData);
	}

	Matrix* Matrix::imap(float (*func)(float)) {
		// Apply function to matrix
		for (size_t row = 0; row < rows; row++) {
			for (size_t col = 0; col < cols; col++) {
				data[row][col] = func(data[row][col]);
			}
		}
		return this;
	}


	Matrix Matrix::scale(float val) {
		// Apply function to matrix
		std::vector<std::vector<float>> newData;
		for (size_t newRow = 0; newRow < rows; newRow++) {
			newData.push_back(std::vector<float>());
			for (size_t newCol = 0; newCol < cols; newCol++) {
				newData[newRow].push_back(data[newRow][newCol] * val);
			}
		}
		return Matrix(newData);
	}

	Matrix* Matrix::iscale(float val) {
		// Apply function to matrix
		for (size_t row = 0; row < rows; row++) {
			for (size_t col = 0; col < cols; col++) {
				data[row][col] = data[row][col] * val;
			}
		}
		return this;
	}


	Matrix Matrix::ewise(Matrix& other, float (*func)(float, float)) {
		// Get variables
		std::vector<std::vector<float>> newData;
		size_t oRows = other.getRows();
		size_t oCols = other.getCols();
		std::vector<std::vector<float>>& oData = other.getData();

		// Add to other matrix
		for (size_t newRow = 0; newRow < rows; newRow++) {
			newData.push_back(std::vector<float>());
			for (size_t newCol = 0; newCol < cols; newCol++) {
				float value = data[newRow][newCol];
				float oValue = oData[newRow < oRows ? newRow : oRows - 1][newCol < oCols ? newCol : oCols - 1];
				newData[newRow].push_back(func(value, oValue));
			}
		}
		return Matrix(newData);
	}

	Matrix* Matrix::iewise(Matrix& other, float (*func)(float, float)) {
		// Get variables
		size_t oRows = other.getRows();
		size_t oCols = other.getCols();
		std::vector<std::vector<float>>& oData = other.getData();

		// Inplace add to other matrix
		for (size_t row = 0; row < rows; row++) {
			for (size_t col = 0; col < cols; col++) {
				float value = data[row][col];
				float oValue = oData[row < oRows ? row : oRows - 1][col < oCols ? col : oCols - 1];
				data[row][col] = func(value, oValue);
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

	void Matrix::printValues(std::string tag) {
		// Print each layers values
		std::cout << tag << std::endl;
		for (size_t row = 0; row < rows; row++) {
			std::cout << "  ";
			for (size_t col = 0; col < cols; col++) {
				char prefix = (data[row][col] >= 0) ? ' ' : '\0';
				std::cout << prefix << std::fixed << std::setprecision(4) << data[row][col] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	void Matrix::printDims(std::string tag) {
		// Print dimensions
		std::cout << tag << rows << " x " << cols << std::endl;
	}


	std::vector<Matrix> Matrix::splitRows(size_t splitSize) {
		// Initialize variables
		std::vector<std::vector<float>>& data = getData();
		size_t splitCount = (int)(rows / splitSize);
		std::vector<Matrix> splits = std::vector<Matrix>(splitCount + ((rows % splitSize == 0) ? 0 : 1));

		// Loop over each split and copy rows
		for (size_t i = 0; i < splitCount; i++) {
			std::vector<std::vector<float>> splitData = std::vector<std::vector<float>>(splitSize);
			for (size_t o = 0; o < splitSize; o++) splitData[o] = data[i * splitSize + o];
			splits[i] = Matrix(splitData);
		}

		// Copy rows for last split
		if ((rows % splitSize) != 0) {
			std::vector<std::vector<float>> splitData = std::vector<std::vector<float>>(rows % splitSize);
			for (size_t o = 0; o < rows % splitSize; o++) splitData[o] = data[splitCount * splitSize + o];
			splits[splitCount] = Matrix(splitData);
		}

		// Return splits
		return splits;
	}

	size_t Matrix::getRows() { return rows; }

	size_t Matrix::getCols() { return cols; }

	std::vector<std::vector<float>>& Matrix::getData() { return data; }

	float Matrix::get(size_t row, size_t col) {
		// Return data, and check if indices are inbound
		if (row >= 0 && col >= 0 && row < rows && col < cols) return data[row][col];
		else throw std::invalid_argument("Index out of range.");
	}

	bool Matrix::getEmpty() { return rows == 0 || cols == 0; }


	void Matrix::clear() {
		// Clear out matrix
		data.clear();
		rows = 0;
		cols = 0;
	}

	Matrix Matrix::copy() {
		// Return new matrix with same data
		std::vector<std::vector<float>> newData;
		for (size_t newRow = 0; newRow < rows; newRow++) {
			std::vector<float> rowData;
			for (size_t newCol = 0; newCol < cols; newCol++) {
				rowData.push_back(data[newRow][newCol]);
			}
			newData.push_back(rowData);
		}
		return Matrix(newData);
	}

	#pragma endregion
}
