#include "stdafx.h"
#include "_Tensor.h"
#include <omp.h>
#include <cassert>

namespace tbml
{
	const _Tensor _Tensor::ZERO = _Tensor();

	_Tensor::_Tensor()
	{
		shape = {};
		data = {};
	}

	_Tensor::_Tensor(const _Tensor& t)
	{
		shape = t.shape;
		data = t.data;
	}

	_Tensor::_Tensor(const std::vector<size_t>& shape, float v)
	{
		this->shape = shape;
		size_t dataSize = 1;
		for (size_t i = 0; i < shape.size(); i++) dataSize *= shape[i];
		data = std::vector<float>(dataSize, v);
	}

	_Tensor::_Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
	{
		this->shape = shape;
		size_t dataSize = 1;
		for (size_t i = 0; i < shape.size(); i++) dataSize *= shape[i];
		assert(dataSize == data.size());
		this->data = data;
	}

	_Tensor::_Tensor(const std::vector<float>& data)
	{
		this->shape = { data.size() };
		this->data = data;
	}

	_Tensor::_Tensor(const std::vector<std::vector<float>>& data)
	{
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

	_Tensor& _Tensor::operator+=(const _Tensor& t)
	{
		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] += t.data[i];
		return *this;
	}

	_Tensor& _Tensor::operator+=(float v)
	{
		for (size_t i = 0; i < data.size(); i++) data[i] += v;
		return *this;
	}

	_Tensor& _Tensor::operator-=(const _Tensor& t)
	{
		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] -= t.data[i];
		return *this;
	}

	_Tensor& _Tensor::operator-=(float v)
	{
		for (size_t i = 0; i < data.size(); i++) data[i] -= v;
		return *this;
	}

	_Tensor& _Tensor::operator*=(const _Tensor& t)
	{
		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] *= t.data[i];
		return *this;
	}

	_Tensor& _Tensor::operator*=(float v)
	{
		for (size_t i = 0; i < data.size(); i++) data[i] *= v;
		return *this;
	}

	_Tensor& _Tensor::operator/=(const _Tensor& t)
	{
		assert(shape == t.shape);
		for (size_t i = 0; i < data.size(); i++) data[i] /= t.data[i];
		return *this;
	}

	_Tensor& _Tensor::operator/=(float v)
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
			assert(shape == t.shape);

			return this->operator*=(t);
		}

		else if (shape.size() == 2)
		{
			assert(shape[1] == t.shape[0]);

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
		assert(shape.size() == 2);

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

	void _Tensor::print(std::string tag) const
	{
		std::cout << tag << std::endl;

		std::string shapeStr;
		for (size_t i = 0; i < shape.size(); i++) shapeStr += std::to_string(shape[i]) + " ";
		std::cout << "\t( " << shapeStr << ")" << std::endl;

		std::string dataStr;
		for (size_t i = 0; i < data.size(); i++) dataStr += std::to_string(data[i]) + " ";
		std::cout << "\t[ " << dataStr << "]" << std::endl;
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
