#include "stdafx.h"
#include "_Tensor.h"

namespace tbml
{
	_Tensor& _Tensor::operator+=(const _Tensor& t)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::operator+=(float v)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::operator-=(const _Tensor& t)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::operator-=(float v)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::operator*=(const _Tensor& t)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::operator*=(float v)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::operator/=(const _Tensor& t)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::operator/=(float v)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::map(std::function<float(float)> fn)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::ewise(const _Tensor& t, std::function<float(float, float)> fn)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::matmul(const _Tensor& t)
	{
		// TODO: insert return statement here
	}

	_Tensor& _Tensor::transpose()
	{
		// TODO: insert return statement here
	}

	float _Tensor::acc(std::function<float(float, float)> fn, float initial) const
	{
		return 0.0f;
	}

	void _Tensor::print(std::string tag) const
	{}

	bool _Tensor::isZero() const
	{
		return false;
	}

	_Tensor::_Tensor()
	{}

	_Tensor::_Tensor(const _Tensor& t)
	{}

	_Tensor::_Tensor(std::vector<size_t> shape)
	{}

	_Tensor::_Tensor(std::vector<size_t> shape, std::vector<float>&& data)
	{}

	_Tensor::_Tensor(std::vector<float>& data)
	{}

	void _Tensor::resize(std::vector<size_t> shape)
	{}

	void _Tensor::zero()
	{}
}
