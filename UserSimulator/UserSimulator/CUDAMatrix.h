#include <stdlib.h>
#include <stdexcept>

#pragma once
class CUDAMatrix
{
private:
	int _rows;
	int _cols;
	double* _underlyingMatrix;
	bool _arrayForm;
public:
	//CUDAMatrix();
	CUDAMatrix(int rows, int cols);
	CUDAMatrix();
	~CUDAMatrix();
	CUDAMatrix(const CUDAMatrix& other);

	CUDAMatrix tanh();
	CUDAMatrix exp();
	CUDAMatrix RowAverage();
	CUDAMatrix RowAverageMatrix();

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
	CUDAMatrix Array();
	void Destroy();
	void Print();

#pragma region operator overloads
	double& operator()(int row, int col);
	CUDAMatrix operator-(CUDAMatrix mat2);
	CUDAMatrix operator+(CUDAMatrix mat2);
	CUDAMatrix operator*(const CUDAMatrix& mat2) const;
	CUDAMatrix operator*(double constValue);
	CUDAMatrix operator/(double constValue);
	CUDAMatrix operator/(const CUDAMatrix& mat2) const;
	CUDAMatrix& operator=(CUDAMatrix mat2);
	CUDAMatrix& operator+=(const CUDAMatrix& mat2);
	CUDAMatrix operator-=(CUDAMatrix mat2);
#pragma endregion

};