#pragma once

#include<functional>
#include<vector>
#include<string>

namespace tbml
{
	class Matrix
	{
	public:
		Matrix();
		Matrix(const Matrix& data);
		Matrix(const std::vector<std::vector<float>>& data);
		Matrix(std::vector<float>&& data, size_t rows, size_t cols);
		Matrix(size_t rows, size_t cols);

		void resize(size_t rows, size_t cols);
		void clear();

		float& operator()(size_t row, size_t col) { return data[row * cols + col]; }
		float operator()(size_t row, size_t col) const { return data[row * cols + col]; }
		Matrix& operator+=(Matrix const& m);
		Matrix& operator+=(float v);
		Matrix& operator-=(Matrix const& m);
		Matrix& operator-=(float v);
		Matrix& operator*=(Matrix const& m);
		Matrix& operator*=(float v);
		Matrix& operator/=(Matrix const& m);
		Matrix& operator/=(float v);
		Matrix operator+(Matrix const& m) const { return Matrix(*this) += m; }
		Matrix operator+(float v) const { return Matrix(*this) += v; }
		Matrix operator-(Matrix const& m) const { return Matrix(*this) -= m; }
		Matrix operator-(float v) const { return Matrix(*this) -= v; }
		Matrix operator*(Matrix const& m) const { return Matrix(*this) *= m; }
		Matrix operator*(float v) const { return Matrix(*this) *= v; }
		Matrix operator/(Matrix const& m) const { return Matrix(*this) /= m; }
		Matrix operator/(float v) const { return Matrix(*this) /= v; }

		Matrix& map(std::function<float(float)> func);
		Matrix& ewise(Matrix const& m, std::function<float(float, float)> func);
		Matrix& transpose();
		Matrix& cross(Matrix const& m);
		Matrix mapped(std::function<float(float)> func) const { return Matrix(*this).map(func); }
		Matrix ewised(Matrix const& m, std::function<float(float, float)> func) const { return Matrix(*this).ewise(m, func); }
		Matrix transposed() const { return Matrix(*this).transpose(); }
		Matrix crossed(Matrix const& m) const { return Matrix(*this).cross(m); }
		float acc(std::function<float(float, float)> func, float initial) const;
		Matrix& addBounded(Matrix const& m);
		void printValues(std::string tag = "Matrix:") const;
		void printDims(std::string tag = "Dimensions: ") const;

		std::vector<float>& getData() { return data; }
		const size_t getRowCount() const { return rows; }
		const size_t getColCount() const { return cols; }
		bool getEmpty() const { return rows == 0 || cols == 0; }
		std::vector<Matrix> groupRows(size_t splitSize) const;

	private:
		std::vector<float> data;
		size_t rows = 0, cols = 0;
	};
}
