
#pragma once

namespace tbml
{
	class Matrix;
	class Matrix
	{
	public:
		Matrix();
		Matrix(std::vector<std::vector<float>> data_);
		Matrix(size_t rows_, size_t cols_);
		void resize(size_t rows_, size_t cols_);
		void clear();

		Matrix cross(const Matrix& other) const;
		Matrix* icross(const Matrix& other);
		Matrix transpose() const;
		Matrix* itranspose();

		float acc(std::function<float(float, float)> func, float initial) const;
		Matrix map(std::function<float(float)> func) const;
		Matrix* imap(std::function<float(float)> func);
		Matrix ewise(const Matrix& other, std::function<float(float, float)> func) const;
		Matrix* iewise(const Matrix& other, std::function<float(float, float)> func);

		Matrix scale(float val) const;
		Matrix* iscale(float val);
		Matrix add(const Matrix& other) const;
		Matrix* iadd(const Matrix& other);
		Matrix sub(const Matrix& other) const;
		Matrix* isub(const Matrix& other);
		Matrix times(const Matrix& other) const;
		Matrix* itimes(const Matrix& other);
		Matrix div(const Matrix& other) const;
		Matrix* idiv(const Matrix& other);

		void printValues(std::string tag = "Matrix:") const;
		void printDims(std::string tag = "Dimensions: ") const;

		std::vector<Matrix> splitRows(size_t splitSize) const;
		std::vector<std::vector<float>>& getData();
		size_t getRows() const;
		size_t getCols() const;
		float get(size_t row, size_t col) const;
		void set(size_t row, size_t col, float val);
		bool getEmpty() const;

	private:
		std::vector<std::vector<float>> data;
		size_t rows, cols;
	};
}
