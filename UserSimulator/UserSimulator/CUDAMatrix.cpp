#include "CUDAMatrix.h"

CUDAMatrix::CUDAMatrix() : _rows(-1), _cols(-1), _underlyingMatrix(nullptr), _arrayForm(false) {}

CUDAMatrix::CUDAMatrix(int rows, int cols) : _rows(rows), _cols(cols), _underlyingMatrix(new double[rows * cols]), _arrayForm(false) {

}

CUDAMatrix::~CUDAMatrix() {
	delete[] _underlyingMatrix;
	_underlyingMatrix = nullptr;
}

// Copy Constructor
//CUDAMatrix::CUDAMatrix(const CUDAMatrix& other) : _rows(other._rows), _cols(other._cols), _underlyingMatrix(new double[other._rows * other._cols]), _arrayForm(other._arrayForm) {
//	if (!_underlyingMatrix) _underlyingMatrix = (double*)malloc(_rows * _cols * sizeof(double));
//	std::copy(other._underlyingMatrix, other._underlyingMatrix + other._rows * other._cols, this->_underlyingMatrix);
//}

CUDAMatrix::CUDAMatrix(const CUDAMatrix& other)
	: _rows(other._rows), _cols(other._cols), _underlyingMatrix(new double[other._rows * other._cols]), _arrayForm(other._arrayForm) {
	std::copy(other._underlyingMatrix, other._underlyingMatrix + other._rows * other._cols, _underlyingMatrix);
}

double& CUDAMatrix::operator()(int row, int col) {
	if (row >= _rows || col >= _cols) {
		throw std::out_of_range("matrix indices out of range");
	}

	return _underlyingMatrix[row * _cols + col];
}

// old version
CUDAMatrix& CUDAMatrix::operator=(CUDAMatrix inputMatrix) {
	if(this == &inputMatrix) return *this;

	this->_arrayForm = false;
	this->_cols = inputMatrix.GetColumns();
	this->_rows = inputMatrix.GetRows();

	delete[] this->_underlyingMatrix;
	_underlyingMatrix = new double[_rows * _cols];

	std::copy(inputMatrix.GetUnderlyingMatrix(), inputMatrix.GetUnderlyingMatrix() + _rows * _cols, this->_underlyingMatrix);

	return *this;

}

void CUDAMatrix::Resize(int rows, int columns) {
	delete[] _underlyingMatrix; // Free the old memory
	//_underlyingMatrix = nullptr;

	_rows = rows;
	_cols = columns;
	_underlyingMatrix = new double[rows * columns];
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
	std::fill(newMatrix._underlyingMatrix, newMatrix._underlyingMatrix + rows * columns, 0.0);
	return newMatrix;
}

CUDAMatrix CUDAMatrix::Array() {
	CUDAMatrix res = *this;
	res._arrayForm = true;
	return res;
}

void CUDAMatrix::Destroy() {
	delete[] _underlyingMatrix;
	_underlyingMatrix = nullptr;
}