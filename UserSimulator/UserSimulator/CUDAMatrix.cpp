#include "CUDAMatrix.h"

CUDAMatrix::CUDAMatrix() {
	//_underlyingMatrix = NULL;
	_underlyingMatrix = nullptr;
	_rows = -1;
	_cols = -1;
	_arrayForm = false;
}

CUDAMatrix::CUDAMatrix(int rows, int cols) {
	_underlyingMatrix = (double*)malloc(cols * rows * sizeof(double));
	_rows = rows;
	_cols = cols;
	_arrayForm = false;
}

double& CUDAMatrix::operator()(int row, int col) {
	if (row >= _rows || col >= _cols) {
		throw std::out_of_range("matrix indices out of range");
	}

	return _underlyingMatrix[row * _cols + col];
	printf("FUCK %f\n", _underlyingMatrix[row * _cols + col]);
}

CUDAMatrix CUDAMatrix::operator=(CUDAMatrix inputMatrix) {
	if(this == &inputMatrix) return *this;

	this->_arrayForm = false;
	this->_cols = inputMatrix.GetColumns();
	this->_rows = inputMatrix.GetRows();
	if (_underlyingMatrix == 0) _underlyingMatrix = (double*)malloc(_rows * _cols * sizeof(double));
	std::copy(inputMatrix.GetUnderlyingMatrix(), inputMatrix.GetUnderlyingMatrix() + _rows * _cols, this->_underlyingMatrix);

	return *this;

}

void CUDAMatrix::Resize(int rows, int columns) {
	_rows = rows;
	_cols = columns;
	//if (_underlyingMatrix) free(_underlyingMatrix);
	_underlyingMatrix = (double*)malloc(_rows * _cols * sizeof(double));
	//_underlyingMatrix = (double*)realloc(_underlyingMatrix, _rows * _cols * sizeof(double));
}

void CUDAMatrix::Print() {
	for (int i = 0; i < _rows; i++) {
		for (int j = 0; j < _cols; j++) {
			printf("%f", _underlyingMatrix[i * _cols + j]);
			if (j < _cols - 1) printf(" ");
		}
		printf("\n");
	}
	printf("\n");
}

CUDAMatrix CUDAMatrix::Zero(int rows, int columns) {
	CUDAMatrix newMatrix(rows, columns);
	memset(newMatrix._underlyingMatrix, 0, newMatrix._rows * newMatrix._cols * sizeof(double));
	return newMatrix;
}

CUDAMatrix CUDAMatrix::Array() {
	// TODO OPTIMIZE
	//_arrayForm = true;
	//return *this;

	CUDAMatrix res(this->_rows, this->_cols);
	res._arrayForm = true;
	res.SetUnderlyingMatrix(this->GetUnderlyingMatrix());
	return res;
}

void CUDAMatrix::Destroy() {
	if(_underlyingMatrix) free(_underlyingMatrix);
}