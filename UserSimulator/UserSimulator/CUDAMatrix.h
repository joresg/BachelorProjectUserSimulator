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
	CUDAMatrix();
	CUDAMatrix(int rows, int cols);

	void tanh();

	int GetRows() { return _rows; }
	int GetColumns() { return _cols; }
	void SetRows(int rows) { _rows = rows; }
	void SetColumns(int columns) { _cols = columns; }
	double* GetUnderlyingMatrix() { return _underlyingMatrix; }
	void SetUnderlyingMatrix(double* mat) { _underlyingMatrix = mat; }
	void Resize(int rows, int columns);
	static CUDAMatrix Zero(int rows, int columns);
	CUDAMatrix Array();
	void Destroy();
	void Print();

#pragma region operator overloads
	double& operator()(int row, int col);
	CUDAMatrix operator-(CUDAMatrix mat2);
	CUDAMatrix operator+(CUDAMatrix mat2);
	CUDAMatrix operator*(CUDAMatrix mat2);
	CUDAMatrix operator*(double constValue);
	CUDAMatrix operator=(CUDAMatrix mat2);
#pragma endregion

};