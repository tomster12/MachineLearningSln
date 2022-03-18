
#pragma once

#include <vector>
#include <string>


class Matrix {
private:
	// Private variables
	std::vector<std::vector<float>> data;
	size_t rows, cols;

public:
	// Public constructor
	Matrix();
	Matrix(std::vector<std::vector<float>> data_);
	Matrix(size_t rows_, size_t cols_);

	// Public arithmetic
	Matrix cross(Matrix other);
	Matrix* icross(Matrix other);
	Matrix transpose();
	Matrix* itranspose();

	float acc(float (*func)(float, float), float initial);
	Matrix map(float (*func)(float));
	Matrix* imap(float (*func)(float));
	Matrix scale(float val);
	Matrix* iscale(float val);

	Matrix ewise(Matrix other, float (*func)(float, float));
	Matrix* iewise(Matrix other, float (*func)(float, float));
	Matrix add(Matrix other);
	Matrix* iadd(Matrix other);
	Matrix sub(Matrix other);
	Matrix* isub(Matrix other);
	Matrix times(Matrix other);
	Matrix* itimes(Matrix other);
	Matrix div(Matrix other);
	Matrix* idiv(Matrix other);
	
	
	// Public functions
	void printValues(std::string tag="Matrix:");
	void printDims(std::string tag="Dimensions: ");
	
	std::vector<Matrix> splitRows(size_t splitSize);
	size_t getRows();
	size_t getCols();
	std::vector<std::vector<float>>* getData();
	float get(size_t row, size_t col);
	bool getEmpty();

	Matrix copy();
};
