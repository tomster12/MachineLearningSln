#pragma once

namespace tbml
{
	// Column-major order vector<float> based tensor
	class _Tensor
	{
	public:
		static const _Tensor ZERO;

		_Tensor();
		_Tensor(const _Tensor& t);
		_Tensor(const std::vector<size_t>& shape, float v);
		_Tensor(const std::vector<size_t>& shape, const std::vector<float>& data);
		_Tensor(const std::vector<float>& data);
		_Tensor(const std::vector<std::vector<float>>& data);
		_Tensor(const std::vector<std::vector<std::vector<float>>>& data);
		void zero();

		template<typename... Args>
		float& operator()(Args... args) { return data[_getIndex(0, 1, args...)]; }

		template<typename... Args>
		float operator()(Args... args) const { return data[_getIndex(0, 1, args...)]; }

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

		float acc(std::function<float(float, float)> fn, float initial) const;
		_Tensor& map(std::function<float(float)> fn);
		_Tensor& ewise(const _Tensor& t, std::function<float(float, float)> fn);
		_Tensor& matmul(const _Tensor& t);
		_Tensor& transpose();
		_Tensor mapped(std::function<float(float)> fn) const { return _Tensor(*this).map(fn); }
		_Tensor ewised(const _Tensor& t, std::function<float(float, float)> fn) const { return _Tensor(*this).ewise(t, fn); }
		_Tensor matmulled(const _Tensor& t) const { return _Tensor(*this).matmul(t); }
		_Tensor transposed() const { return _Tensor(*this).transpose(); }

		void print(std::string tag = "Tensor:") const;
		const std::vector<size_t> getShape() const { return shape; }
		const std::vector<float>& getData() const { return data; }
		bool isZero() const;

	private:
		std::vector<size_t> shape;
		std::vector<float> data;

		template<typename ICurrent, typename... IRest>
		size_t _getIndex(size_t acc, size_t mult, ICurrent index, IRest... rest) const
		{
			return _getIndex(acc + (index * mult), (mult * shape[shape.size() - sizeof...(IRest)]), rest...);
		}

		size_t _getIndex(size_t acc, size_t mult) const { return acc; }
	};
}
