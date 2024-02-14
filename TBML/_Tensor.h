#pragma once

namespace tbml
{
	class _Tensor
	{
	public:
		_Tensor();
		_Tensor(const _Tensor& t);
		_Tensor(std::vector<size_t> shape);
		_Tensor(std::vector<size_t> shape, std::vector<float>&& data);

		_Tensor(std::vector<float>& data);
		_Tensor(std::vector<std::vector<float>>& data);
		_Tensor(std::vector<std::vector<std::vector<float>>>& data);

		void resize(std::vector<size_t> shape);
		void zero();

		_Tensor& operator+=(const _Tensor& t);
		_Tensor& operator+=(float v);
		_Tensor& operator-=(const _Tensor& t);
		_Tensor& operator-=(float v);
		_Tensor& operator*=(const _Tensor& t);
		_Tensor& operator*=(float v);
		_Tensor& operator/=(const _Tensor& t);
		_Tensor& operator/=(float v);
		_Tensor operator+(const _Tensor& t) const { return _Tensor(*this) += t; }
		_Tensor operator+(float v) const { return _Tensor(*this) += v; }
		_Tensor operator-(const _Tensor& t) const { return _Tensor(*this) -= t; }
		_Tensor operator-(float v) const { return _Tensor(*this) -= v; }
		_Tensor operator*(const _Tensor& t) const { return _Tensor(*this) *= t; }
		_Tensor operator*(float v) const { return _Tensor(*this) *= v; }
		_Tensor operator/(const _Tensor& t) const { return _Tensor(*this) /= t; }
		_Tensor operator/(float v) const { return _Tensor(*this) /= v; }
		_Tensor& map(std::function<float(float)> fn);
		_Tensor& ewise(const _Tensor& t, std::function<float(float, float)> fn);
		_Tensor& matmul(const _Tensor& t);
		_Tensor& transpose();
		float acc(std::function<float(float, float)> fn, float initial) const;

		void print(std::string tag = "_Tensor:") const;
		const std::vector<size_t> getShape() const { return shape; }
		const std::vector<float>& getData() { return data; }
		bool isZero() const;

	private:
		std::vector<size_t> shape;
		std::vector<float> data;
	};
}
