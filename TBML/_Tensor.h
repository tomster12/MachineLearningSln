#pragma once

namespace tbml
{
	// Column-major order vector<float> based tensor
	// e.g. shape[0] = rows, shape[1] = columns, ...
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
		float& at(Args... args) { return data[_getIndex(0, 1, args...)]; }

		template<typename... Args>
		float at(Args... args) const { return data[_getIndex(0, 1, args...)]; }

		template<typename... Args>
		float& operator()(Args... args) { return at(args...); }

		template<typename... Args>
		float operator()(Args... args) const { return at(args...); }

		_Tensor& add(const _Tensor& t);
		_Tensor& add(const _Tensor& t, size_t moddim);
		_Tensor& add(float v);
		_Tensor& sub(const _Tensor& t);
		_Tensor& sub(float v);
		_Tensor& mult(const _Tensor& t);
		_Tensor& mult(float v);
		_Tensor& div(const _Tensor& t);
		_Tensor& div(float v);
		float acc(std::function<float(float, float)> fn, float initial) const;
		_Tensor& map(std::function<float(float)> fn);
		_Tensor& ewise(const _Tensor& t, std::function<float(float, float)> fn);
		_Tensor& matmul(const _Tensor& t);
		_Tensor& transpose();
		_Tensor mapped(std::function<float(float)> fn) const { return _Tensor(*this).map(fn); }
		_Tensor ewised(const _Tensor& t, std::function<float(float, float)> fn) const { return _Tensor(*this).ewise(t, fn); }
		_Tensor matmulled(const _Tensor& t) const { return _Tensor(*this).matmul(t); }
		_Tensor transposed() const { return _Tensor(*this).transpose(); }

		_Tensor& operator+=(const _Tensor& t) { return add(t); }
		_Tensor& operator+=(float v) { return add(v); }
		_Tensor& operator-=(const _Tensor& t) { return sub(t); }
		_Tensor& operator-=(float v) { return sub(v); }
		_Tensor& operator*=(const _Tensor& t) { return mult(t); }
		_Tensor& operator*=(float v) { return mult(v); }
		_Tensor& operator/=(const _Tensor& t) { return div(t); }
		_Tensor& operator/=(float v) { return div(v); }
		_Tensor operator+(const _Tensor& t) const { return _Tensor(*this).add(t); }
		_Tensor operator+(float v) const { return _Tensor(*this).add(v); }
		_Tensor operator-(const _Tensor& t) const { return _Tensor(*this).sub(t); }
		_Tensor operator-(float v) const { return _Tensor(*this).sub(v); }
		_Tensor operator*(const _Tensor& t) const { return _Tensor(*this).mult(t); }
		_Tensor operator*(float v) const { return _Tensor(*this).mult(v); }
		_Tensor operator/(const _Tensor& t) const { return _Tensor(*this).div(t); }
		_Tensor operator/(float v) const { return _Tensor(*this).div(v); }

		void print(std::string tag = "Tensor:") const;
		std::vector<_Tensor> groupRows(size_t targetGroupSize) const;
		const std::vector<size_t> getShape() const { return shape; }
		const size_t getShape(size_t dim) const { return dim <= shape.size() ? shape[dim] : 1; }
		const size_t getDims() const { return shape.size(); }
		const std::vector<float>& getData() const { return data; }
		bool isZero() const;

	private:
		std::vector<size_t> shape;
		std::vector<float> data;

		template<typename ICurrent, typename... IRest>
		size_t _getIndex(size_t acc, size_t mult, ICurrent index, IRest... rest) const
		{
			// t[a, b, c] = data[a + b * shape[0] + c * shape[0] * shape[1]]
			return _getIndex(acc + (index * mult), (mult * shape[shape.size() - sizeof...(IRest) - 1]), rest...);
		}

		size_t _getIndex(size_t acc, size_t mult) const { return acc; }
	};
}
