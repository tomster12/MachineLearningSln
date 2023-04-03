
#pragma once


namespace tbml
{

	class Matrix
	{
	private:
		std::vector<std::vector<float>> data;
		size_t rows, cols;

	public:
		Matrix();
		Matrix(std::vector<std::vector<float>> data_);
		Matrix(size_t rows_, size_t cols_);

		Matrix cross(Matrix& other);
		Matrix* icross(Matrix& other);
		Matrix transpose();
		Matrix* itranspose();

		float acc(float (*func)(float, float), float initial);
		Matrix map(float (*func)(float));
		Matrix* imap(float (*func)(float));
		Matrix ewise(Matrix& other, float (*func)(float, float));
		Matrix* iewise(Matrix& other, float (*func)(float, float));

		Matrix scale(float val);
		Matrix* iscale(float val);
		Matrix add(Matrix& other);
		Matrix* iadd(Matrix& other);
		Matrix sub(Matrix& other);
		Matrix* isub(Matrix& other);
		Matrix times(Matrix& other);
		Matrix* itimes(Matrix& other);
		Matrix div(Matrix& other);
		Matrix* idiv(Matrix& other);

		void printValues(std::string tag = "Matrix:");
		void printDims(std::string tag = "Dimensions: ");

		std::vector<Matrix> splitRows(size_t splitSize);
		std::vector<std::vector<float>>& getData();
		size_t getRows();
		size_t getCols();
		float get(size_t row, size_t col);
		bool getEmpty();

		void clear();
		Matrix copy();
	};
}