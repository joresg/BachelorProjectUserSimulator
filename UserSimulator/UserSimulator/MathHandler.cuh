#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDAMatrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <math.h>
#include <stdexcept>

#pragma once
class MathHandler {
public:
	MathHandler(unsigned long randomSeed);
	CUDAMatrix TransposeMatrix(CUDAMatrix inputMatrix);
	double GenerateRandomNumber(double min, double max);
	CUDAMatrix TanhDerivative(CUDAMatrix inputMatrix);
	std::default_random_engine GetRandomEngine() { return _re; }
private:
	unsigned long _randSeed;
	std::uniform_real_distribution<double> _unif;
	std::default_random_engine _re;
	dim3 _blockSize;
};

//cudaError_t addWithCuda(double* c, const double* a, const double* b, unsigned int size);