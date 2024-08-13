#include <stdlib.h>
#include <stdexcept>

#pragma once
enum LayerActivationFuncs {
	reluAct,
	tanhAct,
	sigAct,
	leakyReLU
};

class CUDAMatrix
{
private:
	int _rows;
	int _cols;
	double* _underlyingMatrix;
	bool _arrayForm;
	bool _vectorForm;
public:
	//CUDAMatrix();
	CUDAMatrix(int rows, int cols);
	CUDAMatrix();
	~CUDAMatrix();
	CUDAMatrix(const CUDAMatrix& other);

	CUDAMatrix Activate(LayerActivationFuncs fn);
	CUDAMatrix tanh();
	CUDAMatrix sigmoid();
	CUDAMatrix exp();
	CUDAMatrix transpose();
	CUDAMatrix RowAverage();

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