#include <stdlib.h>
#include <stdexcept>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#pragma once
enum LayerActivationFuncs {
	reluAct,
	tanhAct,
	sigAct,
	leakyReLU
};

enum GatedUnits {
	NoGates,
	GRU
};

class CUDAMatrix
{
private:
	friend class boost::serialization::access;

	int _rows;
	int _cols;
	double* _underlyingMatrix;
	bool _arrayForm;
	bool _vectorForm;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& _rows;
		ar& _cols;
		ar& _arrayForm;
		ar& _vectorForm;

		if (Archive::is_saving::value) {
			std::vector<double> temp(_underlyingMatrix, _underlyingMatrix + _rows * _cols);
			ar& temp;
		}

		if (Archive::is_loading::value) {
			std::vector<double> temp;
			ar& temp;
			if (_underlyingMatrix) delete[] _underlyingMatrix;
			_underlyingMatrix = new double[_rows * _cols];
			std::copy(temp.begin(), temp.end(), _underlyingMatrix);
		}
	}
public:
	//CUDAMatrix();
	CUDAMatrix(int rows, int cols);
	CUDAMatrix();
	~CUDAMatrix();
	CUDAMatrix(const CUDAMatrix& other);

	CUDAMatrix Activate(LayerActivationFuncs fn);
	CUDAMatrix ApplyMask(CUDAMatrix mask);
	CUDAMatrix tanh();
	CUDAMatrix sigmoid();
	CUDAMatrix exp();
	CUDAMatrix log();
	CUDAMatrix transpose();
	CUDAMatrix sqrt();
	CUDAMatrix RowAverage();
	// Frobenius norm for gradient clipping
	double Norm();
	CUDAMatrix ClipByNorm(double thresholdMin, double thresholdMax);

	int GetRows() const { return _rows; }
	int GetColumns() const { return _cols; }
	void SetRows(int rows) { _rows = rows; }
	void SetColumns(int columns) { _cols = columns; }
	double* GetUnderlyingMatrix() const { return _underlyingMatrix; }
	void SetUnderlyingMatrix(double* mat) {
		delete[] _underlyingMatrix;
		_underlyingMatrix = mat;
	}
	void Resize(int rows, int columns);
	static CUDAMatrix Zero(int rows, int columns);
	static CUDAMatrix One(int rows, int columns);
	CUDAMatrix Array();
	CUDAMatrix Vec();
	void Destroy();
	void Print();

#pragma region operator overloads
	double& operator()(int row, int col);
	CUDAMatrix operator-(CUDAMatrix mat2);
	CUDAMatrix operator+(CUDAMatrix mat2);
	CUDAMatrix operator+(double constValue);
	CUDAMatrix operator-(double constValue);
	CUDAMatrix operator*(const CUDAMatrix& mat2) const;
	CUDAMatrix operator*(double constValue);
	CUDAMatrix operator/(double constValue);
	CUDAMatrix operator/(const CUDAMatrix& mat2) const;
	CUDAMatrix& operator=(CUDAMatrix mat2);
	CUDAMatrix& operator+=(const CUDAMatrix& mat2);
	CUDAMatrix operator-=(CUDAMatrix mat2);
#pragma endregion

};