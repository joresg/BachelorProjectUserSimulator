//#include "cuda_runtime.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "CUDAMatrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <math.h>
#include <stdexcept>
#include <queue>
#include <chrono>
#include <vector>

//#include <cuda_runtime.h>
//#include <algorithm>
//#include <vector>

#pragma once
class MathHandler {
public:
	MathHandler(unsigned long randomSeed);
	CUDAMatrix TransposeMatrix(CUDAMatrix inputMatrix);
	double GenerateRandomNumber(double min, double max);
	CUDAMatrix TanhDerivative(CUDAMatrix inputMatrix);
	std::default_random_engine GetRandomEngine() { return _re; }
	std::vector<CUDAMatrix> CreateOneHotEncodedVector(std::vector<int> cmdIDs, int allClasses);
	std::vector<CUDAMatrix> CreateBatchOneHotEncodedVector(std::vector<std::vector<int>> cmdIDs, int allClasses, int batchSize);
private:
	unsigned long _randSeed;
	std::uniform_real_distribution<double> _unif;
	std::default_random_engine _re;
	dim3 _blockSize;
};