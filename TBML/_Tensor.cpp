#include "stdafx.h"
#include "_Tensor.h"
#include <omp.h>
#include <cassert>

namespace tbml
{
	const _Tensor _Tensor::ZERO = _Tensor();

	_Tensor::_Tensor()
	{
		// Default constructor
		shape = {};
		data = {};
	}

	_Tensor::_Tensor(const _Tensor& t)
	{
		// Copy constructor
		shape = t.shape;
		data = t.data;
	}

	_Tensor::_Tensor(const std::vector<size_t>& shape, float v)
	{
		// Create tensor with shape and fill with v
		this->shape = shape;
		size_t dataSize = 1;
		for (size_t i = 0; i < shape.size(); i++) dataSize *= shape[i];
		data = std::vector<float>(dataSize, v);
	}

	_Tensor::_Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
	{
		// Create tensor with shape and data and assert data fits
		this->shape = shape;
		size_t dataSize = 1;
		for (size_t i = 0; i < shape.size(); i++) dataSize *= shape[i];
		assert(dataSize == data.size());
		this->data = data;
	}

	_Tensor::_Tensor(const std::vector<float>& data)
	{
		// Create 1D tensor
		this->shape = { data.size() };
		this->data = data;
	}

	_Tensor::_Tensor(const std::vector<std::vector<float>>& data)
	{
		// Create 2D tensor
		shape = { data.size(), data[0].size() };
		this->data = std::vector<float>(shape[0] * shape[1]);
		for (size_t row = 0; row < shape[0]; row++)
		{
			for (size_t col = 0; col < shape[1]; col++)
			{
				this->data[row + col * shape[0]] = data[row][col];
			}
		}
	}

	_Tensor::_Tensor(const std::vector<std::vector<std::vector<float>>>& data)
	{
		// Create 3D tensor
		shape = { data[0].size(), data[0][0].size(), data.size() };
		this->data = std::vector<float>(shape[0] * shape[1] * shape[2]);
		for (size_t x = 0; x < shape[0]; x++)
		{
			for (size_t y = 0; y < shape[1]; y++)
			{
				for (size_t z = 0; z < shape[2]; z++)
				{
					this->data[x + y * shape[0] + z * shape[0] * shape[1]] = data[z][x][y];
				}
			}
		}
	}

	void _Tensor::zero()
	{
		for (size_t i = 0; i < data.size(); i++) data[i] = 0;
	}

	_Tensor& _Tensor::add(const _Tensor& t)
	{
		if (shape.size() == 0)
		{
			shape = t.shape;
			data = t.data;
			return *this;
		}
	
		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] += t.data[i];
		return *this;
	}

	_Tensor& _Tensor::add(const _Tensor& t, size_t moddim)
	{
		// TODO: Figure out the more generic way to do this
		assert(moddim < 2);
		for (size_t i = 0; i < shape.size(); i++)
		{
			if (i != moddim) assert(shape[i] == t.shape[i]);
		}

		// shape = (3, 4, 2)
		// [ 0, 3, 6, 9  ] .. [ 12, 15, 18, 21 ]
		// [ 1, 4, 7, 10 ] .. [ 13, 16, 19, 22 ]
		// [ 2, 5, 8, 11 ] .. [ 14, 17, 20, 23 ]

		if (moddim == 0)
		{
			// shape = (1, 4, 2) => Take all the data to closest row 0
			// [ 0, 1, 2, 3 ] .. [ 4, 5, 6, 7 ]
			// [ 0, 1, 2, 3 ] .. [ 4, 5, 6, 7 ]
			// [ 0, 1, 2, 3 ] .. [ 4, 5, 6, 7 ]
			// ni = i // 3

			for (size_t i = 0; i < data.size(); i++)
			{
				int ni = (int)(i / shape[0]);
				data[i] += t.data[ni];
			}
		}

		else if (moddim == 1)
		{
			// shape = (3, 1, 2) => Take all the data to closest col 0
			// [ 0, 0, 0, 0 ] .. [ 3, 3, 3, 3 ]
			// [ 1, 1, 1, 1 ] .. [ 4, 4, 4, 4 ]
			// [ 2, 2, 2, 2 ] .. [ 5, 5, 5, 5 ]

			for (size_t i = 0; i < data.size(); i++)
			{
				int ni = (int)(i / (shape[0] * shape[1])) + (i % shape[0]);
				data[i] += t.data[ni];
			}
		}

		return *this;
	}

	_Tensor& _Tensor::add(float v)
	{
		for (size_t i = 0; i < data.size(); i++) data[i] += v;
		return *this;
	}

	_Tensor& _Tensor::sub(const _Tensor& t)
	{
		if (shape.size() == 0)
		{
			shape = t.shape;
			data = t.data;
			for (size_t i = 0; i < data.size(); i++) data[i] = -data[i];
			return *this;
		}

		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] -= t.data[i];
		return *this;
	}

	_Tensor& _Tensor::sub(float v)
	{
		for (size_t i = 0; i < data.size(); i++) data[i] -= v;
		return *this;
	}

	_Tensor& _Tensor::mult(const _Tensor& t)
	{
		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] *= t.data[i];
		return *this;
	}

	_Tensor& _Tensor::mult(float v)
	{
		for (size_t i = 0; i < data.size(); i++) data[i] *= v;
		return *this;
	}

	_Tensor& _Tensor::div(const _Tensor& t)
	{
		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] /= t.data[i];
		return *this;
	}

	_Tensor& _Tensor::div(float v)
	{
		for (size_t i = 0; i < data.size(); i++) data[i] /= v;
		return *this;
	}

	float _Tensor::acc(std::function<float(float, float)> fn, float initial) const
	{
		float acc = initial;
		for (size_t i = 0; i < data.size(); i++) acc = fn(data[i], acc);
		return acc;
	}

	_Tensor& _Tensor::map(std::function<float(float)> fn)
	{
		for (size_t i = 0; i < data.size(); i++) data[i] = fn(data[i]);
		return *this;
	}

	_Tensor& _Tensor::ewise(const _Tensor& t, std::function<float(float, float)> fn)
	{
		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] = fn(data[i], t.data[i]);
		return *this;
	}

	_Tensor& _Tensor::matmul(const _Tensor& t)
	{
		if (shape.size() == 1)
		{
			assert(getShape(0) == t.getShape(0));

			return this->operator*=(t);
		}

		else if (shape.size() == 2)
		{
			assert(getShape(1) == t.getShape(0));

			const std::vector<float>& a = data;
			const std::vector<float>& b = t.data;
			std::vector<float> result(shape[0] * t.shape[1]);

			#pragma omp parallel for num_threads(4)
			for (int row = 0; row < (int)shape[0]; row++)
			{
				for (int ocol = 0; ocol < (int)t.shape[1]; ocol++)
				{
					for (int i = 0; i < (int)shape[1]; i++)
					{
						// TODO: See if can use operator()
						result[row + shape[0] * ocol] += a[row + shape[0] * i] * b[i + t.shape[0] * ocol];
					}
				}
			}

			data = std::move(result);
			shape[1] = t.shape[1];
			return *this;
		}

		throw std::runtime_error("Invalid shape for matrix multiplication");
	}

	_Tensor& _Tensor::transpose()
	{
		if (shape.size() == 1)
		{
			shape = { 1, shape[0] };
			return *this;
		}

		else if (shape.size() == 2)
		{
			std::vector<float> result(shape[0] * shape[1]);

			for (size_t row = 0; row < shape[0]; row++)
			{
				for (size_t col = 0; col < shape[1]; col++)
				{
					result[col + shape[1] * row] = data[row + shape[0] * col];
				}
			}

			data = std::move(result);
			shape = { shape[1], shape[0] };
			return *this;
		}

		throw std::runtime_error("Transpose not defined for dim > 2");
	}

	void _Tensor::print(std::string tag) const
	{
		std::cout << tag << std::endl;

		std::string shapeStr;
		for (size_t i = 0; i < shape.size(); i++) shapeStr += std::to_string(shape[i]) + " ";
		std::cout << "\t( " << shapeStr << ")" << std::endl;

		std::string dataStr;

		if (getDims() == 1)
		{
			dataStr += "\t[ ";
			for (size_t i = 0; i < data.size(); i++) dataStr += std::to_string(data[i]) + " ";
			dataStr += "]";
		}

		else if (getDims() == 2)
		{
			for (size_t x = 0; x < shape[0]; x++)
			{
				dataStr += "\t[ ";
				for (size_t y = 0; y < shape[1]; y++)
				{
					dataStr += std::to_string(data[x + shape[0] * y]) + " ";
				}
				dataStr += "]\n";
			}
		}

		std::cout << dataStr << std::endl;
	}

	bool _Tensor::isZero() const
	{
		for (size_t i = 0; i < shape.size(); i++)
		{
			if (shape[i] != 0) return false;
		}
		return true;
	}
}
